from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaia_max.domain import (
    AnswerType,
    IntegerAnswer,
    Modality,
    OutputContract,
    Question,
    RiskLevel,
    TaskClass,
    TaskProfile,
    TemporalConstraint,
)
from gaia_max.output_contracts import ContractParseStatus
from gaia_max.profiles import TaskProfileRecord, load_profile_registry
from gaia_max.solver_routing import (
    DeterministicRoutePolicy,
    RouteDecisionSource,
    RouteFailureCode,
    SolverCatalog,
    SolverRegistry,
    SolverRoute,
    SolverRoutingError,
)
from gaia_max.task_analysis import AnalysisAuthority, ReconciledTaskAnalysis

SHIPPED_REGISTRY = Path("config/task_profiles.json")


def analysis(
    *,
    task_class: TaskClass = TaskClass.UNKNOWN,
    modality: Modality = Modality.WEB,
    operation: str = "research the requested fact",
    temporal: TemporalConstraint | None = None,
    profile_used: bool = False,
) -> ReconciledTaskAnalysis:
    authority = AnalysisAuthority.PROFILE if profile_used else AnalysisAuthority.MODEL
    return ReconciledTaskAnalysis(
        task_id="routing-task",
        modality=modality,
        task_class=task_class,
        requested_operation=operation,
        temporal_constraint=temporal,
        output_contract=OutputContract(answer_type=AnswerType.INTEGER),
        field_sources={
            "modality": authority,
            "task_class": authority,
            "requested_operation": authority,
            "temporal_constraint": authority,
            "filters": authority,
            "output_contract": authority,
            "risk_flags": authority,
        },
        contract_parse_status=ContractParseStatus.COMPLETE,
        profile_used=profile_used,
        model_used=not profile_used,
        model_provider="routing-test-model" if not profile_used else None,
        model_confidence=0.95 if not profile_used else None,
        requires_review=not profile_used,
    )


def profiled_case(
    record: TaskProfileRecord,
) -> tuple[Question, TaskProfile, ReconciledTaskAnalysis]:
    runtime_question = Question(
        task_id=record.task_id,
        question=f"Synthetic public question for {record.task_id}",
        file_name=record.expected_file_name,
    )
    task_profile = TaskProfile(
        task_id=record.task_id,
        question=runtime_question.question,
        question_sha256="a" * 64,
        file_name=runtime_question.file_name,
        modality=record.modality,
        task_class=record.task_class,
        route=record.route,
        risk_level=record.risk_level,
        temporal_constraint=record.temporal_constraint,
        requested_operation=record.requested_operation,
        required_sources=record.required_sources,
        output_contract=record.output_contract,
        verification_policy=record.verification_policy,
        risk_flags=record.risk_flags,
    )
    reconciled = ReconciledTaskAnalysis(
        task_id=record.task_id,
        modality=record.modality,
        task_class=record.task_class,
        requested_operation=record.requested_operation,
        temporal_constraint=record.temporal_constraint,
        output_contract=record.output_contract,
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
    return runtime_question, task_profile, reconciled


def test_catalog_defines_every_stable_route_exactly_once() -> None:
    catalog = SolverCatalog()

    assert set(catalog.routes) == set(SolverRoute)
    assert len(catalog.routes) == len(SolverRoute)


def test_every_shipped_profile_selects_its_reviewed_route() -> None:
    registry = load_profile_registry(SHIPPED_REGISTRY)
    policy = DeterministicRoutePolicy()

    decisions = []
    for record in registry.profiles:
        question, profile, reconciled = profiled_case(record)
        decisions.append(policy.decide(question, reconciled, profile=profile))

    assert len(decisions) == registry.profile_count == 20
    assert [decision.route.value for decision in decisions] == [
        record.route for record in registry.profiles
    ]
    assert all(not decision.fallback_used for decision in decisions)


@pytest.mark.parametrize(
    ("file_name", "task_class", "modality", "expected"),
    [
        ("program.py", TaskClass.CODE, Modality.PYTHON, SolverRoute.CODE),
        ("book.xlsx", TaskClass.SPREADSHEET, Modality.XLSX, SolverRoute.SPREADSHEET),
        ("data.csv", TaskClass.SPREADSHEET, Modality.XLSX, SolverRoute.SPREADSHEET),
        (
            "recipe.mp3",
            TaskClass.AUDIO_INGREDIENT,
            Modality.AUDIO,
            SolverRoute.AUDIO_INGREDIENT,
        ),
        (
            "pages.wav",
            TaskClass.AUDIO_NUMERIC,
            Modality.AUDIO,
            SolverRoute.AUDIO_NUMERIC,
        ),
        ("board.png", TaskClass.CHESS, Modality.IMAGE, SolverRoute.CHESS),
    ],
)
def test_attachment_extensions_route_to_modality_specialists(
    file_name: str,
    task_class: TaskClass,
    modality: Modality,
    expected: SolverRoute,
) -> None:
    question = Question(
        task_id="routing-task",
        question="Process this official attachment.",
        file_name=file_name,
    )

    decision = DeterministicRoutePolicy().decide(
        question,
        analysis(task_class=task_class, modality=modality),
    )

    assert decision.route is expected
    assert decision.fallback_used is False


@pytest.mark.parametrize(
    ("wording", "expected"),
    [
        (
            "In https://youtube.com/watch?v=test, what is the maximum number "
            "of species visible at the same time?",
            SolverRoute.YOUTUBE_VISUAL,
        ),
        (
            "At https://youtu.be/test, what exact words did the speaker say in reply?",
            SolverRoute.YOUTUBE_SPEECH,
        ),
    ],
)
def test_youtube_wording_is_a_hard_signal(wording: str, expected: SolverRoute) -> None:
    decision = DeterministicRoutePolicy().decide(
        Question(task_id="routing-task", question=wording),
        analysis(),
    )

    assert decision.route is expected
    assert decision.source is RouteDecisionSource.HARD_SIGNAL


def test_conflicting_youtube_intents_block_instead_of_guessing() -> None:
    question = Question(
        task_id="routing-task",
        question=(
            "In https://youtube.com/watch?v=test, inspect the visible frame and return "
            "the exact words said by the speaker."
        ),
    )

    with pytest.raises(SolverRoutingError) as caught:
        DeterministicRoutePolicy().decide(question, analysis())

    assert caught.value.code is RouteFailureCode.HARD_SIGNAL_CONFLICT


def test_historical_wikipedia_constraint_routes_to_revision_solver() -> None:
    temporal = TemporalConstraint(
        kind="revision_cutoff",
        end=datetime(2020, 1, 1, tzinfo=UTC),
    )
    question = Question(
        task_id="routing-task",
        question="Using the Wikipedia page as it existed by the cutoff, count the entries.",
    )

    decision = DeterministicRoutePolicy().decide(
        question,
        analysis(temporal=temporal),
    )

    assert decision.route is SolverRoute.WIKIPEDIA_HISTORY
    assert decision.source is RouteDecisionSource.HARD_SIGNAL


@pytest.mark.parametrize(
    ("task_class", "operation", "expected"),
    [
        (
            TaskClass.OPERATION_TABLE,
            "check every ordered pair in the operation table",
            SolverRoute.OPERATION_TABLE,
        ),
        (
            TaskClass.STRUCTURED_TABLE,
            "find the maximum row and return another column",
            SolverRoute.STRUCTURED_TABLE,
        ),
        (
            TaskClass.SCHOLARLY,
            "follow the article to its primary paper",
            SolverRoute.SCHOLARLY,
        ),
    ],
)
def test_task_shapes_choose_matching_capability(
    task_class: TaskClass,
    operation: str,
    expected: SolverRoute,
) -> None:
    decision = DeterministicRoutePolicy().decide(
        Question(task_id="routing-task", question=operation),
        analysis(task_class=task_class, operation=operation),
    )

    assert decision.route is expected


def test_unknown_web_task_uses_bounded_deep_research_fallback() -> None:
    decision = DeterministicRoutePolicy().decide(
        Question(task_id="routing-task", question="Identify this obscure public fact."),
        analysis(),
    )

    assert decision.route is SolverRoute.DEEP_RESEARCH_FALLBACK
    assert decision.source is RouteDecisionSource.DEEP_RESEARCH_FALLBACK
    assert decision.fallback_used is True


def test_reviewed_deep_research_profile_is_a_valid_explicit_fallback() -> None:
    question = Question(task_id="routing-task", question="Research this obscure fact.")
    profile = TaskProfile(
        task_id=question.task_id,
        question=question.question,
        question_sha256="a" * 64,
        modality=Modality.WEB,
        task_class=TaskClass.DEEP_RESEARCH,
        route=SolverRoute.DEEP_RESEARCH_FALLBACK.value,
        risk_level=RiskLevel.MEDIUM,
        requested_operation="research the requested fact",
        output_contract=OutputContract(answer_type=AnswerType.INTEGER),
        verification_policy="claim-level evidence",
    )

    decision = DeterministicRoutePolicy().decide(
        question,
        analysis(task_class=TaskClass.DEEP_RESEARCH, profile_used=True),
        profile=profile,
    )

    assert decision.route is SolverRoute.DEEP_RESEARCH_FALLBACK
    assert decision.source is RouteDecisionSource.DEEP_RESEARCH_FALLBACK
    assert decision.fallback_used is True


def test_unknown_audio_task_cannot_hide_behind_web_research() -> None:
    with pytest.raises(SolverRoutingError) as caught:
        DeterministicRoutePolicy().decide(
            Question(task_id="routing-task", question="Interpret this audio."),
            analysis(modality=Modality.AUDIO),
        )

    assert caught.value.code is RouteFailureCode.NO_SAFE_FALLBACK


def test_ambiguous_audio_attachment_blocks_instead_of_choosing_a_subtype() -> None:
    with pytest.raises(SolverRoutingError) as caught:
        DeterministicRoutePolicy().decide(
            Question(
                task_id="routing-task",
                question="Interpret this recording.",
                file_name="recording.mp3",
            ),
            analysis(task_class=TaskClass.UNKNOWN, modality=Modality.AUDIO),
        )

    assert caught.value.code is RouteFailureCode.AMBIGUOUS_CAPABILITY
    assert set(caught.value.detail_codes) == {"audio_ingredient", "audio_numeric"}


def test_unknown_or_unsupported_attachment_blocks() -> None:
    with pytest.raises(SolverRoutingError) as caught:
        DeterministicRoutePolicy().decide(
            Question(
                task_id="routing-task",
                question="Inspect this attachment.",
                file_name="document.pdf",
            ),
            analysis(),
        )

    assert caught.value.code is RouteFailureCode.UNSUPPORTED_ATTACHMENT
    assert caught.value.detail_codes == (".pdf",)


def test_profile_cannot_override_conflicting_python_attachment() -> None:
    question = Question(
        task_id="routing-task",
        question="Run this program.",
        file_name="program.py",
    )
    task_profile = TaskProfile(
        task_id=question.task_id,
        question=question.question,
        question_sha256="a" * 64,
        file_name=question.file_name,
        modality=Modality.XLSX,
        task_class=TaskClass.SPREADSHEET,
        route=SolverRoute.SPREADSHEET.value,
        risk_level=RiskLevel.LOW,
        requested_operation="calculate workbook totals",
        output_contract=OutputContract(answer_type=AnswerType.INTEGER),
        verification_policy="two_path_parity",
    )

    with pytest.raises(SolverRoutingError) as caught:
        DeterministicRoutePolicy().decide(
            question,
            analysis(
                task_class=TaskClass.SPREADSHEET,
                modality=Modality.XLSX,
                profile_used=True,
            ),
            profile=task_profile,
        )

    assert caught.value.code is RouteFailureCode.PROFILE_ROUTE_CONFLICT


class RecordingSolver:
    def __init__(self) -> None:
        self.routes: list[str] = []

    async def solve(
        self,
        _question: Question,
        _analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> object:
        self.routes.append(route)
        return IntegerAnswer(value=7)


@pytest.mark.asyncio
async def test_registry_dispatches_only_to_explicitly_registered_solver() -> None:
    registry = SolverRegistry()
    solver = RecordingSolver()
    registry.register(SolverRoute.STRUCTURED_TABLE, solver)

    result = await registry.solve(
        SolverRoute.STRUCTURED_TABLE,
        Question(task_id="routing-task", question="Count rows."),
        analysis(task_class=TaskClass.STRUCTURED_TABLE),
    )

    assert result == IntegerAnswer(value=7)
    assert solver.routes == [SolverRoute.STRUCTURED_TABLE.value]


def test_registry_rejects_duplicate_registration() -> None:
    registry = SolverRegistry()
    registry.register(SolverRoute.CODE, RecordingSolver())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(SolverRoute.CODE, RecordingSolver())


@pytest.mark.asyncio
async def test_registry_reports_missing_solver_without_substitution() -> None:
    with pytest.raises(SolverRoutingError) as caught:
        await SolverRegistry().solve(
            SolverRoute.CHESS,
            Question(task_id="routing-task", question="Find the best move."),
            analysis(task_class=TaskClass.CHESS, modality=Modality.IMAGE),
        )

    assert caught.value.code is RouteFailureCode.SOLVER_NOT_REGISTERED
