from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from smolagents import ChatMessage, ChatMessageToolCall, Model
from smolagents.models import ChatMessageToolCallFunction, MessageRole

from gaia_max.domain import (
    AnswerType,
    Modality,
    OutputContract,
    Question,
    RiskLevel,
    SemanticAnswer,
    TaskClass,
    TaskProfile,
    TaskStatus,
    Verdict,
    VerificationResult,
)
from gaia_max.observability import OpenTelemetry
from gaia_max.research.composition import build_deep_research_components
from gaia_max.research.models import IndexableDocument, ResearchStatus
from gaia_max.snapshot import question_sha256
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis
from gaia_max.task_graph import TaskGraphInput, build_task_graph


class ResearchScriptModel(Model):
    def __init__(self, passage_id: str) -> None:
        super().__init__(model_id="research-integration-script")
        self.passage_id = passage_id
        self.calls = 0

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        del messages, stop_sequences, response_format, kwargs
        self.calls += 1
        if tools_to_call_from is None:
            return ChatMessage(role=MessageRole.ASSISTANT, content="Retrieve then cite.")
        if self.calls == 2:
            tool_name = next(
                tool.name for tool in tools_to_call_from if tool.name != "final_answer"
            )
            return _tool_call(
                tool_name,
                {"payload_json": '{"query":"official recipient Ada","top_k":1}'},
                "retrieve",
            )
        output = json.dumps(
            {
                "semantic_answer": {"kind": "string", "value": "Ada"},
                "claims": [
                    {
                        "claim_id": "claim_recipient",
                        "statement": "Ada is the official recipient.",
                        "citation_ids": [self.passage_id],
                    }
                ],
                "unresolved_claim_ids": [],
            }
        )
        return _tool_call("final_answer", {"answer": output}, "finish")


def _tool_call(name: str, arguments: dict[str, object], call_id: str) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
                id=call_id,
                type="function",
            )
        ],
    )


class ApprovingOuterVerifier:
    async def verify(
        self,
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        _answer: SemanticAnswer,
    ) -> VerificationResult:
        return VerificationResult(
            verifier="independent-outer-verifier",
            tier=3,
            verdict=Verdict.APPROVE,
            passed_checks=["inner_evidence_gate_completed"],
        )


@pytest.mark.asyncio
async def test_langgraph_smolagents_llamaindex_and_otel_work_as_one_guarded_path() -> None:
    exporter = InMemorySpanExporter()
    telemetry = OpenTelemetry.from_exporter(exporter, service_name="gaia-integration")
    text = (
        "The official award register states that Ada is the official recipient for 2025. "
        "The record was signed by the registrar and published in the annual register."
    )
    source = IndexableDocument(
        document_id="official-award-register",
        text=text,
        model_text_artifact_id="b" * 64,
        source_url="https://official.example/award-register",
        page_number=7,
        retrieved_at=datetime.now(UTC),
        is_primary_source=True,
        safety_disposition="clean",
    )

    # Determine the stable content-addressed citation produced by the same real index.
    preliminary = build_deep_research_components(
        model=object(),
        documents=[source],
        telemetry=telemetry,
    )
    hit = (await preliminary.retriever.retrieve("official recipient Ada", top_k=1))[0]
    components = build_deep_research_components(
        model=ResearchScriptModel(hit.passage_id),
        documents=[source],
        max_agent_steps=4,
        planning_interval=4,
        max_replans=0,
        max_tool_calls=1,
        telemetry=telemetry,
    )
    question = Question(
        task_id="research-integration",
        question="Return the official recipient's name as a string.",
    )
    profile = TaskProfile(
        task_id=question.task_id,
        question=question.question,
        question_sha256=question_sha256(question),
        modality=Modality.WEB,
        task_class=TaskClass.DEEP_RESEARCH,
        route=SolverRoute.DEEP_RESEARCH_FALLBACK.value,
        risk_level=RiskLevel.LOW,
        requested_operation="identify the official recipient",
        required_sources=["official award register"],
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        verification_policy="claim-level primary evidence",
    )
    registry = SolverRegistry()
    registry.register(SolverRoute.DEEP_RESEARCH_FALLBACK, components.solver)
    graph = build_task_graph(
        solver_registry=registry,
        verifier=ApprovingOuterVerifier(),
        telemetry=telemetry,
    )

    state = await graph.ainvoke(TaskGraphInput(question=question, profile=profile).initial_state())
    telemetry.shutdown()

    assert state["status"] is TaskStatus.READY
    assert state["serialized_answer"] == "Ada"
    record = components.ledger.get(question.task_id)
    assert record is not None and record.status is ResearchStatus.APPROVED
    assert len(components.gateway.audit_events) == 1
    span_names = {span.name for span in exporter.get_finished_spans()}
    assert {
        "task_graph.solve",
        "research.solver.run",
        "research.agent.run",
        "tool.invoke",
        "research.retrieve",
    }.issubset(span_names)
    all_attributes = {
        key
        for exported_span in exporter.get_finished_spans()
        for key in exported_span.attributes
    }
    assert not any("answer" in key or "prompt" in key for key in all_attributes)
