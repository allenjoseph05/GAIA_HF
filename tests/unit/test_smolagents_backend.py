from __future__ import annotations

import asyncio
import json

import pytest
from smolagents import ChatMessage, ChatMessageToolCall, Model
from smolagents.models import ChatMessageToolCallFunction, MessageRole

from gaia_max.research.models import ResearchObjective
from gaia_max.research.smolagents_backend import SmolagentsResearchBackend
from gaia_max.retrieval.source_safety import ToolPolicy
from gaia_max.solver_routing import SolverRoute
from gaia_max.tool_enforcement import EnforcedToolGateway, RouteToolPolicy, ToolBudget


class ScriptedToolCallingModel(Model):
    """Offline model script that drives the real smolagents ReAct loop."""

    def __init__(self, final_output: str) -> None:
        super().__init__(model_id="scripted-tool-calling-model")
        self.final_output = final_output
        self.calls = 0

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        del messages, response_format, kwargs
        self.calls += 1
        if tools_to_call_from is None:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Search the approved source, then cite the returned passage.",
            )
        if self.calls == 2:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        function=ChatMessageToolCallFunction(
                            name="search",
                            arguments={"payload_json": '{"query":"recipient"}'},
                        ),
                        id="call-search",
                        type="function",
                    )
                ],
            )
        assert stop_sequences is not None
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(
                        name="final_answer",
                        arguments={"answer": self.final_output},
                    ),
                    id="call-final",
                    type="function",
                )
            ],
        )


@pytest.mark.asyncio
async def test_real_smolagents_loop_calls_only_enforced_tool_and_returns_schema() -> None:
    passage_id = "a" * 64

    async def search(payload):
        assert payload == {"query": "recipient"}
        return {"passages": [{"passage_id": passage_id}]}

    gateway = EnforcedToolGateway(
        policy=RouteToolPolicy(
            route=SolverRoute.DEEP_RESEARCH_FALLBACK,
            tools=ToolPolicy(allowed_tools=("search",)),
            budget=ToolBudget(max_calls=1),
        ),
        handlers={"search": search},
    )
    result_json = json.dumps(
        {
            "semantic_answer": {"kind": "string", "value": "Ada"},
            "claims": [
                {
                    "claim_id": "claim_recipient",
                    "statement": "Ada is the recipient.",
                    "required": True,
                    "citation_ids": [passage_id],
                }
            ],
            "unresolved_claim_ids": [],
        }
    )
    model = ScriptedToolCallingModel(result_json)
    backend = SmolagentsResearchBackend(
        model=model,
        gateway=gateway,
        tool_descriptions={"search": "Search one approved evidence index."},
        max_steps=4,
        planning_interval=4,
    )

    draft = await asyncio.wait_for(
        backend.research(
            ResearchObjective(
                task_id="research-1",
                question="Who is the recipient?",
                required_output='{"answer_type":"string"}',
            )
        ),
        timeout=10,
    )

    assert draft.semantic_answer.value == "Ada"
    assert model.calls == 3
    assert [event.tool_name for event in gateway.audit_events] == ["search"]
