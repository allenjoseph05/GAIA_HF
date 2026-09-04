"""Pure reversed-instruction solver for the deterministic-text route."""

from __future__ import annotations

import re

from gaia_max.domain import Modality, Question, StringAnswer, TaskClass
from gaia_max.solver_routing import SolverRegistry, SolverRoute
from gaia_max.task_analysis import ReconciledTaskAnalysis


class TransformedTextSolverError(ValueError):
    """Base failure for an unsupported or invalid transformed-text task."""


class UnsupportedTransformInstructionError(TransformedTextSolverError):
    """The decoded instruction is outside this deliberately narrow solver."""


_OPPOSITE_WORD = re.compile(
    r"\bwrite\s+(?:the\s+)?opposite\s+of\s+(?:the\s+)?word\s+"
    r"(?P<quote>['\"])(?P<word>[A-Za-z]+)(?P=quote)\s+as\s+the\s+answer\b",
    re.IGNORECASE,
)

_DIRECTION_VECTORS = {
    "down": (0, -1),
    "east": (1, 0),
    "left": (-1, 0),
    "north": (0, 1),
    "right": (1, 0),
    "south": (0, -1),
    "up": (0, 1),
    "west": (-1, 0),
}


def reverse_exact(text: str) -> str:
    """Reverse Unicode code points without trimming or normalizing characters."""

    return text[::-1]


def opposite_direction(word: str) -> str:
    """Return the geometric opposite of a supported direction word."""

    normalized = word.casefold()
    vector = _DIRECTION_VECTORS.get(normalized)
    if vector is None:
        raise UnsupportedTransformInstructionError(
            "decoded opposite-word instruction is not a supported direction"
        )
    opposite_vector = (-vector[0], -vector[1])
    candidates = [
        candidate
        for candidate, candidate_vector in _DIRECTION_VECTORS.items()
        if candidate_vector == opposite_vector
    ]
    family = {"left", "right", "up", "down"} if normalized in {
        "left",
        "right",
        "up",
        "down",
    } else {"north", "south", "east", "west"}
    opposite = next(candidate for candidate in candidates if candidate in family)
    if word.isupper():
        return opposite.upper()
    if word.istitle():
        return opposite.title()
    return opposite


class TransformedTextSolver:
    """Decode a reversed instruction and execute its narrow deterministic operation."""

    async def solve(
        self,
        question: Question,
        analysis: ReconciledTaskAnalysis,
        route: str,
    ) -> object:
        if route != SolverRoute.DETERMINISTIC_TEXT.value:
            raise TransformedTextSolverError("transformed-text solver received wrong route")
        if (
            analysis.task_class is not TaskClass.DETERMINISTIC_TEXT
            or analysis.modality is not Modality.TEXT
            or question.file_name is not None
        ):
            raise TransformedTextSolverError(
                "transformed-text solver received incompatible task metadata"
            )

        decoded = reverse_exact(question.question)
        if reverse_exact(decoded) != question.question:
            raise TransformedTextSolverError("reverse transform failed round-trip invariant")
        match = _OPPOSITE_WORD.search(decoded)
        if match is None:
            raise UnsupportedTransformInstructionError(
                "decoded instruction does not match a supported exact transform"
            )
        return StringAnswer(value=opposite_direction(match.group("word")))


def register_transformed_text_solver(
    registry: SolverRegistry,
    solver: TransformedTextSolver | None = None,
) -> TransformedTextSolver:
    """Register B01 explicitly and return the concrete solver instance."""

    instance = solver or TransformedTextSolver()
    registry.register(SolverRoute.DETERMINISTIC_TEXT, instance)
    return instance
