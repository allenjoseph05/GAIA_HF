from __future__ import annotations

import pytest

from gaia_max.domain import (
    AnswerType,
    Modality,
    OutputContract,
    Question,
    StringAnswer,
    TaskClass,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.solvers.transformed_text import (
    TransformedTextSolver,
    TransformedTextSolverError,
    UnsupportedTransformInstructionError,
    opposite_direction,
    register_transformed_text_solver,
    reverse_exact,
)
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis


def analysis(
    *,
    task_class: TaskClass = TaskClass.DETERMINISTIC_TEXT,
    modality: Modality = Modality.TEXT,
) -> ReconciledTaskAnalysis:
    return ReconciledTaskAnalysis(
        task_id="transform-task",
        modality=modality,
        task_class=task_class,
        requested_operation="exact string transform",
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        risk_flags=("preserve_exact_characters",),
        field_sources={
            "modality": AnalysisAuthority.PROFILE,
            "task_class": AnalysisAuthority.PROFILE,
            "requested_operation": AnalysisAuthority.PROFILE,
            "temporal_constraint": AnalysisAuthority.PROFILE,
            "filters": AnalysisAuthority.PROFILE,
            "output_contract": AnalysisAuthority.PROFILE,
            "risk_flags": AnalysisAuthority.PROFILE,
        },
        contract_parse_status=ContractParseStatus.COMPLETE,
        profile_used=True,
        model_used=False,
        requires_review=False,
    )


@pytest.mark.parametrize(
    "text",
    [
        "ASCII punctuation: !?,.",
        "Accents stay exact: café déjà vu",
        "Emoji and lines 🙂\nsecond line",
        "combining: e\u0301",
        " leading and trailing whitespace ",
        "",
    ],
)
def test_reverse_transform_round_trips_every_exact_character(text: str) -> None:
    transformed = reverse_exact(text)

    assert reverse_exact(transformed) == text
    assert len(transformed) == len(text)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("north", "south"),
        ("south", "north"),
        ("east", "west"),
        ("west", "east"),
        ("up", "down"),
        ("down", "up"),
        ("LEFT", "RIGHT"),
        ("Right", "Left"),
    ],
)
def test_opposite_direction_uses_geometry_and_preserves_simple_case(
    word: str,
    expected: str,
) -> None:
    assert opposite_direction(word) == expected


def test_non_direction_antonym_is_rejected_instead_of_guessed() -> None:
    with pytest.raises(UnsupportedTransformInstructionError):
        opposite_direction("happy")


@pytest.mark.asyncio
async def test_solver_decodes_reversed_instruction_and_returns_typed_answer() -> None:
    decoded = (
        'If you understand this sentence, write the opposite of the word "north" '
        "as the answer."
    )
    question = Question(
        task_id="transform-task",
        question=reverse_exact(decoded),
    )

    result = await TransformedTextSolver().solve(
        question,
        analysis(),
        SolverRoute.DETERMINISTIC_TEXT.value,
    )

    assert result == StringAnswer(value="south")


@pytest.mark.asyncio
async def test_registered_solver_dispatches_through_existing_registry() -> None:
    decoded = 'Write the opposite of the word "up" as the answer.'
    question = Question(task_id="transform-task", question=reverse_exact(decoded))
    registry = SolverRegistry()
    registered = register_transformed_text_solver(registry)

    result = await registry.solve(
        SolverRoute.DETERMINISTIC_TEXT,
        question,
        analysis(),
    )

    assert isinstance(registered, TransformedTextSolver)
    assert result == StringAnswer(value="down")


@pytest.mark.asyncio
async def test_unsupported_decoded_instruction_fails_loudly() -> None:
    decoded = 'Write the opposite of the word "happy" as the answer.'
    question = Question(task_id="transform-task", question=reverse_exact(decoded))

    with pytest.raises(UnsupportedTransformInstructionError):
        await TransformedTextSolver().solve(
            question,
            analysis(),
            SolverRoute.DETERMINISTIC_TEXT.value,
        )


@pytest.mark.asyncio
async def test_wrong_route_or_metadata_is_rejected() -> None:
    decoded = 'Write the opposite of the word "east" as the answer.'
    question = Question(task_id="transform-task", question=reverse_exact(decoded))
    solver = TransformedTextSolver()

    with pytest.raises(TransformedTextSolverError, match="wrong route"):
        await solver.solve(question, analysis(), SolverRoute.STRUCTURED_TABLE.value)
    with pytest.raises(TransformedTextSolverError, match="metadata"):
        await solver.solve(
            question,
            analysis(task_class=TaskClass.STRUCTURED_TABLE, modality=Modality.WEB),
            SolverRoute.DETERMINISTIC_TEXT.value,
        )
