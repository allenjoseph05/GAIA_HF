"""Deterministic capability routing and explicit specialist registration."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePath
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.domain import Modality, Question, TaskClass, TaskProfile
from gaia_max.task_analysis import ReconciledTaskAnalysis


class SolverRoute(StrEnum):
    """Stable route names used by reviewed profiles and specialist milestones."""

    WIKIPEDIA_HISTORY = "wikipedia_history"
    YOUTUBE_VISUAL = "youtube_visual"
    DETERMINISTIC_TEXT = "deterministic_text"
    CHESS = "chess"
    HISTORICAL_WEB = "historical_web"
    OPERATION_TABLE = "operation_table"
    YOUTUBE_SPEECH = "youtube_speech"
    BOTANICAL_CLASSIFICATION = "botanical_classification"
    AUDIO_INGREDIENT = "audio_ingredient"
    CROSSLINGUAL = "crosslingual"
    CODE = "code"
    STRUCTURED_TABLE = "structured_table"
    AUDIO_NUMERIC = "audio_numeric"
    SCHOLARLY = "scholarly"
    CROSSLINGUAL_HISTORICAL = "crosslingual_historical"
    SPREADSHEET = "spreadsheet"
    HISTORICAL_STRUCTURED = "historical_structured"
    DEEP_RESEARCH_FALLBACK = "deep_research_fallback"


class RouteDecisionSource(StrEnum):
    """Strongest source responsible for a route decision."""

    HARD_SIGNAL = "hard_signal"
    REVIEWED_PROFILE = "reviewed_profile"
    CAPABILITY_SCORE = "capability_score"
    DEEP_RESEARCH_FALLBACK = "deep_research_fallback"


class RouteSignalCode(StrEnum):
    """Answer-free facts observed by the deterministic route policy."""

    ATTACHMENT_EXTENSION = "attachment_extension"
    YOUTUBE_VISUAL_INTENT = "youtube_visual_intent"
    YOUTUBE_SPEECH_INTENT = "youtube_speech_intent"
    HISTORICAL_WIKIPEDIA = "historical_wikipedia"
    OPERATION_TABLE_SHAPE = "operation_table_shape"
    STRUCTURED_TABLE_SHAPE = "structured_table_shape"
    TASK_CLASS = "task_class"
    MODALITY = "modality"
    TEMPORAL_CONSTRAINT = "temporal_constraint"
    REVIEWED_PROFILE = "reviewed_profile"


class RouteFailureCode(StrEnum):
    """Stable failure causes suitable for graph errors and repair policy."""

    HARD_SIGNAL_CONFLICT = "hard_signal_conflict"
    PROFILE_ROUTE_UNKNOWN = "profile_route_unknown"
    PROFILE_ROUTE_CONFLICT = "profile_route_conflict"
    UNSUPPORTED_ATTACHMENT = "unsupported_attachment"
    AMBIGUOUS_CAPABILITY = "ambiguous_capability"
    NO_SAFE_FALLBACK = "no_safe_fallback"
    SOLVER_NOT_REGISTERED = "solver_not_registered"


class RouteSignal(BaseModel):
    """One deterministic observation used by routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: RouteSignalCode
    value: str = Field(min_length=1, max_length=200)
    allowed_routes: tuple[SolverRoute, ...] = ()


class RouteDecision(BaseModel):
    """Inspectable, answer-free result of deterministic routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: SolverRoute
    source: RouteDecisionSource
    signals: tuple[RouteSignal, ...]
    candidate_scores: dict[SolverRoute, int]
    profile_route: SolverRoute | None = None
    fallback_used: bool = False

    @model_validator(mode="after")
    def validate_fallback(self) -> RouteDecision:
        is_fallback = self.route is SolverRoute.DEEP_RESEARCH_FALLBACK
        if self.fallback_used != is_fallback:
            raise ValueError("fallback_used must agree with the selected route")
        if is_fallback and self.source is not RouteDecisionSource.DEEP_RESEARCH_FALLBACK:
            raise ValueError("deep-research fallback requires the fallback decision source")
        return self


class SolverCapability(BaseModel):
    """Static description of one route, independent of any solver instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: SolverRoute
    task_classes: tuple[TaskClass, ...]
    modalities: tuple[Modality, ...]
    attachment_extensions: tuple[str, ...] = ()
    requires_temporal_constraint: bool = False
    profile_only: bool = False
    fallback: bool = False

    @model_validator(mode="after")
    def validate_definition(self) -> SolverCapability:
        if self.fallback != (self.route is SolverRoute.DEEP_RESEARCH_FALLBACK):
            raise ValueError("only deep_research_fallback may be marked as fallback")
        normalized = tuple(extension.casefold() for extension in self.attachment_extensions)
        if normalized != self.attachment_extensions:
            raise ValueError("attachment extensions must be lowercase")
        if any(not extension.startswith(".") for extension in normalized):
            raise ValueError("attachment extensions must begin with a dot")
        return self


class SolverRoutingError(RuntimeError):
    """A deterministic route could not be selected or executed safely."""

    def __init__(
        self,
        code: RouteFailureCode,
        message: str,
        *,
        detail_codes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail_codes = detail_codes


class SpecialistSolver(Protocol):
    """Common async boundary implemented by every specialist solver."""

    async def solve(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> object: ...


_AUDIO_EXTENSIONS = (".flac", ".m4a", ".mp3", ".ogg", ".wav")
_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp")
_SPREADSHEET_EXTENSIONS = (".csv", ".xls", ".xlsx")


DEFAULT_SOLVER_CAPABILITIES: tuple[SolverCapability, ...] = (
    SolverCapability(
        route=SolverRoute.WIKIPEDIA_HISTORY,
        task_classes=(TaskClass.HISTORICAL_WIKIPEDIA,),
        modalities=(Modality.WEB,),
        requires_temporal_constraint=True,
    ),
    SolverCapability(
        route=SolverRoute.YOUTUBE_VISUAL,
        task_classes=(TaskClass.YOUTUBE_VISUAL,),
        modalities=(Modality.YOUTUBE_VISUAL,),
    ),
    SolverCapability(
        route=SolverRoute.DETERMINISTIC_TEXT,
        task_classes=(TaskClass.DETERMINISTIC_TEXT,),
        modalities=(Modality.TEXT,),
    ),
    SolverCapability(
        route=SolverRoute.CHESS,
        task_classes=(TaskClass.CHESS,),
        modalities=(Modality.IMAGE,),
        attachment_extensions=_IMAGE_EXTENSIONS,
    ),
    SolverCapability(
        route=SolverRoute.HISTORICAL_WEB,
        task_classes=(TaskClass.HISTORICAL_WEB,),
        modalities=(Modality.WEB,),
    ),
    SolverCapability(
        route=SolverRoute.OPERATION_TABLE,
        task_classes=(TaskClass.OPERATION_TABLE,),
        modalities=(Modality.TEXT, Modality.WEB),
    ),
    SolverCapability(
        route=SolverRoute.YOUTUBE_SPEECH,
        task_classes=(TaskClass.YOUTUBE_SPEECH,),
        modalities=(Modality.YOUTUBE_AUDIO,),
    ),
    SolverCapability(
        route=SolverRoute.BOTANICAL_CLASSIFICATION,
        task_classes=(TaskClass.DEEP_RESEARCH,),
        modalities=(Modality.WEB,),
        profile_only=True,
    ),
    SolverCapability(
        route=SolverRoute.AUDIO_INGREDIENT,
        task_classes=(TaskClass.AUDIO_INGREDIENT,),
        modalities=(Modality.AUDIO,),
        attachment_extensions=_AUDIO_EXTENSIONS,
    ),
    SolverCapability(
        route=SolverRoute.CROSSLINGUAL,
        task_classes=(TaskClass.CROSSLINGUAL,),
        modalities=(Modality.WEB,),
    ),
    SolverCapability(
        route=SolverRoute.CODE,
        task_classes=(TaskClass.CODE,),
        modalities=(Modality.PYTHON,),
        attachment_extensions=(".py",),
    ),
    SolverCapability(
        route=SolverRoute.STRUCTURED_TABLE,
        task_classes=(TaskClass.STRUCTURED_TABLE,),
        modalities=(Modality.WEB, Modality.TEXT),
    ),
    SolverCapability(
        route=SolverRoute.AUDIO_NUMERIC,
        task_classes=(TaskClass.AUDIO_NUMERIC,),
        modalities=(Modality.AUDIO,),
        attachment_extensions=_AUDIO_EXTENSIONS,
    ),
    SolverCapability(
        route=SolverRoute.SCHOLARLY,
        task_classes=(TaskClass.SCHOLARLY,),
        modalities=(Modality.WEB,),
    ),
    SolverCapability(
        route=SolverRoute.CROSSLINGUAL_HISTORICAL,
        task_classes=(TaskClass.CROSSLINGUAL,),
        modalities=(Modality.WEB,),
        requires_temporal_constraint=True,
        profile_only=True,
    ),
    SolverCapability(
        route=SolverRoute.SPREADSHEET,
        task_classes=(TaskClass.SPREADSHEET,),
        modalities=(Modality.XLSX,),
        attachment_extensions=_SPREADSHEET_EXTENSIONS,
    ),
    SolverCapability(
        route=SolverRoute.HISTORICAL_STRUCTURED,
        task_classes=(TaskClass.HISTORICAL_WEB,),
        modalities=(Modality.WEB,),
        requires_temporal_constraint=True,
        profile_only=True,
    ),
    SolverCapability(
        route=SolverRoute.DEEP_RESEARCH_FALLBACK,
        task_classes=(TaskClass.UNKNOWN, TaskClass.DEEP_RESEARCH),
        modalities=(Modality.TEXT, Modality.WEB),
        fallback=True,
    ),
)


class SolverCatalog:
    """Validated immutable lookup of route capability descriptions."""

    def __init__(
        self,
        capabilities: tuple[SolverCapability, ...] = DEFAULT_SOLVER_CAPABILITIES,
    ) -> None:
        by_route = {capability.route: capability for capability in capabilities}
        if len(by_route) != len(capabilities):
            raise ValueError("solver capability routes must be unique")
        if set(by_route) != set(SolverRoute):
            raise ValueError("solver catalog must define every stable route exactly once")
        self._by_route = by_route

    @property
    def routes(self) -> tuple[SolverRoute, ...]:
        return tuple(self._by_route)

    def get(self, route: SolverRoute) -> SolverCapability:
        return self._by_route[route]

    def specialists(self) -> tuple[SolverCapability, ...]:
        return tuple(
            capability
            for capability in self._by_route.values()
            if not capability.fallback
        )


class SolverRegistry:
    """Runtime map containing only real, explicitly registered solver instances."""

    def __init__(self, catalog: SolverCatalog | None = None) -> None:
        self.catalog = catalog or SolverCatalog()
        self._solvers: dict[SolverRoute, SpecialistSolver] = {}

    @property
    def available_routes(self) -> tuple[SolverRoute, ...]:
        return tuple(self._solvers)

    def register(self, route: SolverRoute | str, solver: SpecialistSolver) -> None:
        parsed = SolverRoute(route)
        self.catalog.get(parsed)
        if parsed in self._solvers:
            raise ValueError(f"solver route {parsed.value!r} is already registered")
        if not callable(getattr(solver, "solve", None)):
            raise TypeError("registered solver must provide an async solve boundary")
        self._solvers[parsed] = solver

    def has(self, route: SolverRoute | str) -> bool:
        try:
            parsed = SolverRoute(route)
        except ValueError:
            return False
        return parsed in self._solvers

    async def solve(
        self,
        route: SolverRoute,
        question: Question,
        analysis: ReconciledTaskAnalysis,
    ) -> object:
        solver = self._solvers.get(route)
        if solver is None:
            raise SolverRoutingError(
                RouteFailureCode.SOLVER_NOT_REGISTERED,
                "selected specialist solver is not registered",
                detail_codes=(route.value,),
            )
        return await solver.solve(question, analysis, route.value)


class DeterministicRoutePolicy:
    """Choose specialists by hard constraints, reviewed profiles, then scoring."""

    _youtube_url = re.compile(r"(?:youtube\.com|youtu\.be)", re.IGNORECASE)
    _visual_words = re.compile(
        r"\b(?:appear|frame|shown|simultaneous|species|visible|visual(?:ly)?)\b|same time",
        re.IGNORECASE,
    )
    _speech_words = re.compile(
        r"\b(?:exact words|reply|respond|said|say|says|spoken|speech|transcript|utter)\b",
        re.IGNORECASE,
    )
    _operation_table_words = re.compile(
        r"\b(?:cayley|commutative|noncommutative|operation table|ordered pairs?)\b",
        re.IGNORECASE,
    )
    _structured_table_words = re.compile(
        r"\b(?:argmax|column|highest|lowest|maximum|minimum|row|table)\b",
        re.IGNORECASE,
    )
    _chess_words = re.compile(r"\b(?:best move|chess|fen|position|san)\b", re.IGNORECASE)

    def __init__(self, catalog: SolverCatalog | None = None) -> None:
        self.catalog = catalog or SolverCatalog()

    def decide(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        *,
        profile: TaskProfile | None = None,
    ) -> RouteDecision:
        """Return one deterministic decision or raise a reason-coded routing error."""

        signals = self._signals(question, analysis, profile)
        attachment_signal = next(
            (
                signal
                for signal in signals
                if signal.code is RouteSignalCode.ATTACHMENT_EXTENSION
            ),
            None,
        )
        if attachment_signal is not None and not attachment_signal.allowed_routes:
            raise SolverRoutingError(
                RouteFailureCode.UNSUPPORTED_ATTACHMENT,
                "attachment type has no safe specialist route",
                detail_codes=(attachment_signal.value,),
            )
        hard_route_sets = [
            set(signal.allowed_routes) for signal in signals if signal.allowed_routes
        ]
        hard_allowed = set.intersection(*hard_route_sets) if hard_route_sets else set()
        if hard_route_sets and not hard_allowed:
            raise SolverRoutingError(
                RouteFailureCode.HARD_SIGNAL_CONFLICT,
                "deterministic routing signals disagree",
                detail_codes=tuple(
                    signal.code.value for signal in signals if signal.allowed_routes
                ),
            )

        profile_route = self._profile_route(profile)
        scores = self._score_capabilities(question, analysis, hard_allowed)
        if profile_route is not None:
            capability = self.catalog.get(profile_route)
            if hard_allowed and profile_route not in hard_allowed:
                raise SolverRoutingError(
                    RouteFailureCode.PROFILE_ROUTE_CONFLICT,
                    "reviewed profile route conflicts with a hard routing signal",
                    detail_codes=(profile_route.value,),
                )
            if not self._profile_compatible(capability, question, analysis):
                raise SolverRoutingError(
                    RouteFailureCode.PROFILE_ROUTE_CONFLICT,
                    "reviewed profile route conflicts with reconciled task metadata",
                    detail_codes=(profile_route.value,),
                )
            is_fallback = profile_route is SolverRoute.DEEP_RESEARCH_FALLBACK
            source = (
                RouteDecisionSource.DEEP_RESEARCH_FALLBACK
                if is_fallback
                else (
                    RouteDecisionSource.HARD_SIGNAL
                    if hard_allowed == {profile_route}
                    else RouteDecisionSource.REVIEWED_PROFILE
                )
            )
            return RouteDecision(
                route=profile_route,
                source=source,
                signals=signals,
                candidate_scores=scores,
                profile_route=profile_route,
                fallback_used=is_fallback,
            )

        eligible_scores = {
            route: score
            for route, score in scores.items()
            if route is not SolverRoute.DEEP_RESEARCH_FALLBACK
            and (not hard_allowed or route in hard_allowed)
        }
        best_score = max(eligible_scores.values(), default=0)
        winners = sorted(
            (route for route, score in eligible_scores.items() if score == best_score),
            key=lambda route: route.value,
        )
        if best_score >= 100 and len(winners) == 1:
            route = winners[0]
            source = (
                RouteDecisionSource.HARD_SIGNAL
                if hard_allowed == {route}
                else RouteDecisionSource.CAPABILITY_SCORE
            )
            return RouteDecision(
                route=route,
                source=source,
                signals=signals,
                candidate_scores=scores,
                fallback_used=False,
            )
        if best_score >= 100 and len(winners) > 1:
            raise SolverRoutingError(
                RouteFailureCode.AMBIGUOUS_CAPABILITY,
                "multiple specialist routes have the same strongest capability score",
                detail_codes=tuple(route.value for route in winners),
            )
        if question.file_name is not None:
            raise SolverRoutingError(
                RouteFailureCode.UNSUPPORTED_ATTACHMENT,
                "attachment type has no safe specialist route",
                detail_codes=(self._extension(question.file_name) or "no_extension",),
            )
        if analysis.modality not in {Modality.TEXT, Modality.WEB}:
            raise SolverRoutingError(
                RouteFailureCode.NO_SAFE_FALLBACK,
                "deep research cannot replace a missing modality specialist",
                detail_codes=(analysis.modality.value,),
            )
        return RouteDecision(
            route=SolverRoute.DEEP_RESEARCH_FALLBACK,
            source=RouteDecisionSource.DEEP_RESEARCH_FALLBACK,
            signals=signals,
            candidate_scores=scores,
            fallback_used=True,
        )

    def _signals(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        profile: TaskProfile | None,
    ) -> tuple[RouteSignal, ...]:
        signals: list[RouteSignal] = [
            RouteSignal(code=RouteSignalCode.TASK_CLASS, value=analysis.task_class.value),
            RouteSignal(code=RouteSignalCode.MODALITY, value=analysis.modality.value),
        ]
        if profile is not None:
            signals.append(
                RouteSignal(code=RouteSignalCode.REVIEWED_PROFILE, value=profile.route)
            )
        if analysis.temporal_constraint is not None:
            signals.append(
                RouteSignal(
                    code=RouteSignalCode.TEMPORAL_CONSTRAINT,
                    value=analysis.temporal_constraint.kind,
                )
            )

        if question.file_name is not None:
            extension = self._extension(question.file_name)
            allowed = self._attachment_routes(extension, question, analysis)
            signals.append(
                RouteSignal(
                    code=RouteSignalCode.ATTACHMENT_EXTENSION,
                    value=extension or "no_extension",
                    allowed_routes=allowed,
                )
            )

        combined_text = f"{question.question} {analysis.requested_operation}"
        youtube_context = bool(self._youtube_url.search(combined_text)) or analysis.modality in {
            Modality.YOUTUBE_AUDIO,
            Modality.YOUTUBE_VISUAL,
        }
        if youtube_context:
            visual = bool(self._visual_words.search(combined_text)) or (
                analysis.task_class is TaskClass.YOUTUBE_VISUAL
            )
            speech = bool(self._speech_words.search(combined_text)) or (
                analysis.task_class is TaskClass.YOUTUBE_SPEECH
            )
            if visual:
                signals.append(
                    RouteSignal(
                        code=RouteSignalCode.YOUTUBE_VISUAL_INTENT,
                        value="visual evidence requested",
                        allowed_routes=(SolverRoute.YOUTUBE_VISUAL,),
                    )
                )
            if speech:
                signals.append(
                    RouteSignal(
                        code=RouteSignalCode.YOUTUBE_SPEECH_INTENT,
                        value="spoken evidence requested",
                        allowed_routes=(SolverRoute.YOUTUBE_SPEECH,),
                    )
                )

        if (
            analysis.task_class is TaskClass.HISTORICAL_WIKIPEDIA
            or (
                "wikipedia" in combined_text.casefold()
                and analysis.temporal_constraint is not None
            )
        ):
            signals.append(
                RouteSignal(
                    code=RouteSignalCode.HISTORICAL_WIKIPEDIA,
                    value="dated Wikipedia state requested",
                    allowed_routes=(SolverRoute.WIKIPEDIA_HISTORY,),
                )
            )
        table_signal_allowed = (
            question.file_name is None
            and not youtube_context
            and analysis.modality in {Modality.TEXT, Modality.WEB}
        )
        if table_signal_allowed and (
            analysis.task_class is TaskClass.OPERATION_TABLE
            or (
                analysis.task_class is TaskClass.UNKNOWN
                and self._operation_table_words.search(combined_text)
            )
        ):
            signals.append(
                RouteSignal(
                    code=RouteSignalCode.OPERATION_TABLE_SHAPE,
                    value="algebraic operation-table shape",
                    allowed_routes=(SolverRoute.OPERATION_TABLE,),
                )
            )
        elif table_signal_allowed and (
            analysis.task_class is TaskClass.STRUCTURED_TABLE
            or (
                analysis.task_class is TaskClass.UNKNOWN
                and self._structured_table_words.search(combined_text)
            )
        ):
            signals.append(
                RouteSignal(
                    code=RouteSignalCode.STRUCTURED_TABLE_SHAPE,
                    value="structured row/column reduction",
                    allowed_routes=(
                        SolverRoute.STRUCTURED_TABLE,
                        SolverRoute.HISTORICAL_STRUCTURED,
                    ),
                )
            )
        return tuple(signals)

    def _score_capabilities(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        hard_allowed: set[SolverRoute],
    ) -> dict[SolverRoute, int]:
        extension = self._extension(question.file_name) if question.file_name else None
        scores: dict[SolverRoute, int] = {}
        for capability in self.catalog.specialists():
            score = 0
            if capability.profile_only:
                scores[capability.route] = -1000
                continue
            if analysis.task_class in capability.task_classes:
                score += 100
            if analysis.modality in capability.modalities:
                score += 30
            if extension and extension in capability.attachment_extensions:
                score += 200
            if capability.requires_temporal_constraint:
                score += 25 if analysis.temporal_constraint is not None else -250
            if hard_allowed:
                score += 500 if capability.route in hard_allowed else -1000
            scores[capability.route] = score
        scores[SolverRoute.DEEP_RESEARCH_FALLBACK] = 0
        return scores

    def _profile_route(self, profile: TaskProfile | None) -> SolverRoute | None:
        if profile is None:
            return None
        try:
            return SolverRoute(profile.route.strip())
        except ValueError as exc:
            raise SolverRoutingError(
                RouteFailureCode.PROFILE_ROUTE_UNKNOWN,
                "reviewed profile names an unknown solver route",
            ) from exc

    @staticmethod
    def _profile_compatible(
        capability: SolverCapability,
        question: Question,
        analysis: ReconciledTaskAnalysis,
    ) -> bool:
        if analysis.task_class not in capability.task_classes:
            return False
        if analysis.modality not in capability.modalities:
            return False
        if capability.requires_temporal_constraint and analysis.temporal_constraint is None:
            return False
        if question.file_name is not None and capability.attachment_extensions:
            return DeterministicRoutePolicy._extension(question.file_name) in (
                capability.attachment_extensions
            )
        return True

    def _attachment_routes(
        self,
        extension: str,
        question: Question,
        analysis: ReconciledTaskAnalysis,
    ) -> tuple[SolverRoute, ...]:
        if extension == ".py":
            return (SolverRoute.CODE,)
        if extension in _SPREADSHEET_EXTENSIONS:
            return (SolverRoute.SPREADSHEET,)
        if extension in _AUDIO_EXTENSIONS:
            return (SolverRoute.AUDIO_INGREDIENT, SolverRoute.AUDIO_NUMERIC)
        if extension in _IMAGE_EXTENSIONS:
            combined_text = f"{question.question} {analysis.requested_operation}"
            if analysis.task_class is TaskClass.CHESS or self._chess_words.search(combined_text):
                return (SolverRoute.CHESS,)
        return ()

    @staticmethod
    def _extension(file_name: str) -> str:
        return PurePath(file_name).suffix.casefold()
