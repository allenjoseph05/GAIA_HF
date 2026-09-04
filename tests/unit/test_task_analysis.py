from __future__ import annotations

from collections.abc import Mapping

import pytest

from gaia_max.domain import (
    AnswerType,
    Modality,
    OutputContract,
    Question,
    TaskClass,
    TaskProfile,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.snapshot import question_sha256
from gaia_max.task_analysis import (
    AnalysisAuthority,
    HardContractConflictError,
    LowConfidenceTaskAnalysisError,
    ProfileAnalysisMismatchError,
    StructuredAnalysisResponseError,
    StructuredTaskAnalysisAdapter,
    TaskAnalysisRequest,
    TaskAnalysisUnavailableError,
    TaskAnalyzer,
)


class FakeBackend:
    def __init__(
        self,
        raw: Mapping[str, object] | None = None,
        *,
        error: Exception | None = None,
        name: str = "fixture-structured-model",
    ) -> None:
        self.raw = raw or valid_proposal()
        self.error = error
        self.backend_name = name
        self.calls = 0
        self.requests: list[TaskAnalysisRequest] = []

    @property
    def name(self) -> str:
        return self.backend_name

    async def generate(self, request: TaskAnalysisRequest) -> Mapping[str, object]:
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.raw


def valid_proposal(**updates: object) -> dict[str, object]:
    proposal: dict[str, object] = {
        "modality": "web",
        "task_class": "structured_table",
        "requested_operation": "count matching records deterministically",
        "filters": {"period": "requested"},
        "output_contract": {"answer_type": "integer"},
        "risk_flags": ["source_completeness"],
        "confidence": 0.9,
    }
    proposal.update(updates)
    return proposal


def integer_question() -> Question:
    return Question(task_id="task-1", question="How many matching records are there?")


def matching_profile(question: Question | None = None) -> TaskProfile:
    resolved_question = question or integer_question()
    return TaskProfile(
        task_id=resolved_question.task_id,
        question=resolved_question.question,
        question_sha256=question_sha256(resolved_question),
        file_name=resolved_question.file_name,
        modality=Modality.WEB,
        task_class=TaskClass.STRUCTURED_TABLE,
        route="structured_table",
        risk_level="medium",
        requested_operation="count matching records deterministically",
        filters={"period": "requested"},
        required_sources=["complete_table"],
        output_contract=OutputContract(answer_type=AnswerType.INTEGER),
        verification_policy="deterministic_count",
        risk_flags=["source_completeness"],
    )


def adapter(backend: FakeBackend) -> StructuredTaskAnalysisAdapter:
    return StructuredTaskAnalysisAdapter(backend)


@pytest.mark.asyncio
async def test_known_profile_requires_no_model_and_uses_hard_contract() -> None:
    question = integer_question()

    result = await TaskAnalyzer().analyze(question, profile=matching_profile(question))

    assert result.modality is Modality.WEB
    assert result.task_class is TaskClass.STRUCTURED_TABLE
    assert result.output_contract.answer_type is AnswerType.INTEGER
    assert result.field_sources["modality"] is AnalysisAuthority.PROFILE
    assert result.field_sources["output_contract"] is AnalysisAuthority.HARD_RULE
    assert result.profile_used is True
    assert result.model_used is False
    assert result.requires_review is False


@pytest.mark.asyncio
async def test_model_disagreements_are_recorded_and_cannot_override_profile_or_rules() -> None:
    question = integer_question()
    backend = FakeBackend(
        valid_proposal(
            modality="audio",
            task_class="audio_numeric",
            requested_operation="transcribe an unrelated clip",
            filters={"different": True},
            output_contract={"answer_type": "string"},
            risk_flags=["new_model_risk"],
        )
    )

    result = await TaskAnalyzer(model_adapter=adapter(backend)).analyze(
        question,
        profile=matching_profile(question),
    )

    assert result.modality is Modality.WEB
    assert result.task_class is TaskClass.STRUCTURED_TABLE
    assert result.output_contract.answer_type is AnswerType.INTEGER
    assert {item.field for item in result.disagreements} >= {
        "modality",
        "task_class",
        "requested_operation",
        "filters",
        "output_contract",
    }
    assert all(item.blocks_submission for item in result.disagreements)
    assert result.requires_review is True
    assert "new_model_risk" in result.risk_flags


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_raw",
    [
        valid_proposal(final_answer="must-not-enter"),
        valid_proposal(filters={"nested": {"gold_answer": "must-not-enter"}}),
        valid_proposal(modality="unsupported-modality"),
        valid_proposal(risk_flags=["NOT A SAFE FLAG"]),
    ],
)
async def test_invalid_or_answer_bearing_model_output_never_enters_state(
    invalid_raw: dict[str, object],
) -> None:
    backend = FakeBackend(invalid_raw)

    with pytest.raises(StructuredAnalysisResponseError) as caught:
        await TaskAnalyzer(model_adapter=adapter(backend)).analyze(integer_question())

    assert "must-not-enter" not in str(caught.value)
    assert "must-not-enter" not in repr(caught.value)


@pytest.mark.asyncio
async def test_backend_exception_is_sanitized() -> None:
    backend = FakeBackend(error=RuntimeError("secret raw provider payload"))

    with pytest.raises(StructuredAnalysisResponseError) as caught:
        await TaskAnalyzer(model_adapter=adapter(backend)).analyze(integer_question())

    assert "secret raw provider payload" not in str(caught.value)


def test_backend_name_must_be_safe_metadata() -> None:
    with pytest.raises(ValueError, match="backend name is invalid"):
        adapter(FakeBackend(name="provider\nsecret detail"))


@pytest.mark.asyncio
async def test_new_model_risk_is_preserved_and_forces_review() -> None:
    question = integer_question()
    backend = FakeBackend(
        valid_proposal(risk_flags=["source_completeness", "model_detected_risk"])
    )

    result = await TaskAnalyzer(model_adapter=adapter(backend)).analyze(
        question,
        profile=matching_profile(question),
    )

    assert "model_detected_risk" in result.risk_flags
    assert result.field_sources["risk_flags"] is AnalysisAuthority.POLICY
    assert any(item.field == "risk_flags" for item in result.disagreements)
    assert result.requires_review is True


@pytest.mark.asyncio
async def test_hard_contract_conflict_blocks_before_model_call() -> None:
    backend = FakeBackend()
    question = Question(
        task_id="conflict-task",
        question="Return a comma-separated list in alphabetical order and ascending order.",
    )

    with pytest.raises(HardContractConflictError):
        await TaskAnalyzer(model_adapter=adapter(backend)).analyze(question)

    assert backend.calls == 0


@pytest.mark.asyncio
async def test_unknown_task_without_model_is_not_silently_guessed() -> None:
    with pytest.raises(TaskAnalysisUnavailableError):
        await TaskAnalyzer().analyze(integer_question())


@pytest.mark.asyncio
async def test_unknown_task_uses_model_metadata_but_keeps_hard_output_contract() -> None:
    backend = FakeBackend(valid_proposal(output_contract={"answer_type": "string"}))

    result = await TaskAnalyzer(model_adapter=adapter(backend)).analyze(integer_question())

    assert result.profile_used is False
    assert result.model_used is True
    assert result.output_contract.answer_type is AnswerType.INTEGER
    assert result.field_sources["output_contract"] is AnalysisAuthority.HARD_RULE
    assert result.disagreements[0].field == "output_contract"
    assert result.requires_review is True


@pytest.mark.asyncio
async def test_unknown_wording_can_use_high_confidence_model_contract_for_dry_review() -> None:
    backend = FakeBackend(valid_proposal(output_contract={"answer_type": "string"}))
    question = Question(task_id="unknown-task", question="Provide the appropriate response.")

    result = await TaskAnalyzer(model_adapter=adapter(backend)).analyze(question)

    assert result.contract_parse_status is ContractParseStatus.AMBIGUOUS
    assert result.output_contract.answer_type is AnswerType.STRING
    assert result.field_sources["output_contract"] is AnalysisAuthority.MODEL
    assert result.requires_review is True


@pytest.mark.asyncio
async def test_low_confidence_unknown_task_is_rejected() -> None:
    backend = FakeBackend(valid_proposal(confidence=0.4))

    with pytest.raises(LowConfidenceTaskAnalysisError):
        await TaskAnalyzer(model_adapter=adapter(backend)).analyze(integer_question())


@pytest.mark.asyncio
async def test_profile_must_match_exact_runtime_question_before_model_call() -> None:
    backend = FakeBackend()
    changed_question = Question(task_id="task-1", question="How many changed records exist?")

    with pytest.raises(ProfileAnalysisMismatchError):
        await TaskAnalyzer(model_adapter=adapter(backend)).analyze(
            changed_question,
            profile=matching_profile(integer_question()),
        )

    assert backend.calls == 0


@pytest.mark.asyncio
async def test_ambiguous_hard_contract_stays_marked_for_review_with_profile() -> None:
    question = Question(
        task_id="currency-task",
        question="Express the answer in USD with two decimal places.",
    )
    profile = matching_profile(question).model_copy(
        update={
            "output_contract": OutputContract(
                answer_type=AnswerType.CURRENCY,
                decimal_places=2,
                units_policy="question_specific",
            )
        }
    )

    result = await TaskAnalyzer().analyze(question, profile=profile)

    assert result.contract_parse_status is ContractParseStatus.AMBIGUOUS
    assert result.output_contract.include_currency_symbol is None
    assert result.requires_review is True


@pytest.mark.asyncio
async def test_model_request_contains_only_public_task_and_contract_metadata() -> None:
    backend = FakeBackend()
    await TaskAnalyzer(model_adapter=adapter(backend)).analyze(integer_question())

    request = backend.requests[0]
    assert set(TaskAnalysisRequest.model_fields) == {
        "task_id",
        "question",
        "file_name",
        "deterministic_contract_status",
        "deterministic_contract",
        "contract_issue_codes",
    }
    assert all(
        forbidden not in TaskAnalysisRequest.model_fields
        for forbidden in (
            "candidate_answer",
            "gold_answer",
            "semantic_answer",
            "serialized_answer",
        )
    )
    assert request.question == integer_question().question
