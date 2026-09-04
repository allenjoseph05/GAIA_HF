"""Chessboard square mapping, dual transcription, legal FEN, engine, and SAN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Protocol

import chess
import chess.engine
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactRef, ArtifactSource, ChessMoveAnswer
from gaia_max.domain.models import SHA256_PATTERN

_PIECES = set("PNBRQKpnbrqk")
_SQUARES = tuple(chess.SQUARE_NAMES)


class ChessSolverError(ValueError):
    """Chess perception, reconciliation, legality, engine, or SAN validation failed."""


class BoardOrientation(StrEnum):
    WHITE_BOTTOM = "white_bottom"
    BLACK_BOTTOM = "black_bottom"


class BoardBounds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)

    @model_validator(mode="after")
    def require_square(self) -> BoardBounds:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("chessboard bounds must have positive area")
        if self.right - self.left != self.bottom - self.top:
            raise ValueError("chessboard bounds must be square")
        if (self.right - self.left) % 8:
            raise ValueError("chessboard side length must divide into eight squares")
        return self


class ChessSquareImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    square: StrictStr = Field(pattern=r"^[a-h][1-8]$")
    artifact_id: str = Field(pattern=SHA256_PATTERN)


class SquareObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    square: StrictStr = Field(pattern=r"^[a-h][1-8]$")
    piece: StrictStr | None = Field(default=None, pattern=r"^[PNBRQKpnbrqk]$")
    confidence: float = Field(default=1, ge=0, le=1)


class BoardTranscription(BaseModel):
    """One sensor's complete 64-square occupancy, including explicit empties."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sensor: StrictStr = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")
    orientation: BoardOrientation
    observations: tuple[SquareObservation, ...] = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def require_complete_board(self) -> BoardTranscription:
        squares = tuple(item.square for item in self.observations)
        if len(set(squares)) != 64 or set(squares) != set(_SQUARES):
            raise ValueError("board transcription must cover every square exactly once")
        return self

    def occupancy(self) -> dict[str, str | None]:
        return {item.square: item.piece for item in self.observations}


class TranscriptionComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sensors: tuple[str, ...] = Field(min_length=2)
    disagreements: tuple[str, ...] = ()


class ValidatedChessPosition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fen: StrictStr = Field(min_length=15)
    side_to_move: StrictStr = Field(pattern=r"^[wb]$")
    sensors: tuple[str, ...] = Field(min_length=2)
    adjudicated_squares: tuple[str, ...] = ()


class EngineLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    move_uci: StrictStr = Field(pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$", min_length=4)
    depth: int = Field(ge=1)
    score_cp: int | None = None
    mate_in: int | None = None

    @model_validator(mode="after")
    def require_score(self) -> EngineLine:
        if self.score_cp is None and self.mate_in is None:
            raise ValueError("engine line requires centipawn or mate score")
        return self


class ChessEnginePort(Protocol):
    def analyse(
        self,
        position: ValidatedChessPosition,
        *,
        depth: int,
        multipv: int,
    ) -> Sequence[EngineLine]: ...


class ChessSolution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    position: ValidatedChessPosition
    answer: ChessMoveAnswer
    engine_lines: tuple[EngineLine, ...] = Field(min_length=2)
    winning_verified: bool


def split_chessboard_image(
    store: ArtifactStore,
    artifact: ArtifactRef,
    *,
    bounds: BoardBounds,
    orientation: BoardOrientation,
) -> tuple[ChessSquareImage, ...]:
    """Crop all squares with explicit orientation-to-coordinate mapping."""

    raw = store.read_bytes(artifact)
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if bounds.right > image.width or bounds.bottom > image.height:
                raise ChessSolverError("chessboard bounds exceed image dimensions")
            size = (bounds.right - bounds.left) // 8
            squares: list[ChessSquareImage] = []
            for visual_row in range(8):
                for visual_column in range(8):
                    square = _visual_square(visual_row, visual_column, orientation)
                    crop = image.crop(
                        (
                            bounds.left + visual_column * size,
                            bounds.top + visual_row * size,
                            bounds.left + (visual_column + 1) * size,
                            bounds.top + (visual_row + 1) * size,
                        )
                    )
                    buffer = BytesIO()
                    crop.save(buffer, format="PNG")
                    stored = store.put_bytes(
                        buffer.getvalue(),
                        original_name=f"chess-{square}.png",
                        media_type="image/png",
                        source=ArtifactSource.DERIVED,
                        source_locator=f"artifact:{artifact.artifact_id}#square={square}",
                    )
                    squares.append(
                        ChessSquareImage(square=square, artifact_id=stored.artifact_id)
                    )
            return tuple(squares)
    except ChessSolverError:
        raise
    except Exception as exc:
        raise ChessSolverError("chess image could not be decoded") from exc


def transcription_from_piece_map(
    sensor: str,
    pieces: Mapping[str, str],
    *,
    orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
) -> BoardTranscription:
    if any(square not in _SQUARES for square in pieces):
        raise ChessSolverError("piece map contains an invalid square")
    if any(piece not in _PIECES for piece in pieces.values()):
        raise ChessSolverError("piece map contains an invalid piece symbol")
    return BoardTranscription(
        sensor=sensor,
        orientation=orientation,
        observations=tuple(
            SquareObservation(square=square, piece=pieces.get(square)) for square in _SQUARES
        ),
    )


def compare_transcriptions(
    transcriptions: Sequence[BoardTranscription],
) -> TranscriptionComparison:
    if len(transcriptions) < 2:
        raise ChessSolverError("chess reconciliation requires two transcriptions")
    sensors = tuple(item.sensor for item in transcriptions)
    if len(set(sensors)) != len(sensors):
        raise ChessSolverError("chess transcription sensors must be independent identities")
    orientations = {item.orientation for item in transcriptions}
    if len(orientations) != 1:
        raise ChessSolverError("chess transcriptions disagree on board orientation")
    occupancies = [item.occupancy() for item in transcriptions]
    disagreements = tuple(
        square
        for square in _SQUARES
        if len({occupancy[square] for occupancy in occupancies}) > 1
    )
    return TranscriptionComparison(sensors=sensors, disagreements=disagreements)


def reconcile_chess_position(
    transcriptions: Sequence[BoardTranscription],
    *,
    side_to_move: str,
    adjudicated: Mapping[str, str | None] | None = None,
    castling: str = "-",
    en_passant: str = "-",
) -> ValidatedChessPosition:
    comparison = compare_transcriptions(transcriptions)
    if side_to_move not in {"w", "b"}:
        raise ChessSolverError("chess side to move must be w or b")
    resolutions = dict(adjudicated or {})
    if set(resolutions) != set(comparison.disagreements):
        raise ChessSolverError("every and only disputed square must be adjudicated")
    if any(piece is not None and piece not in _PIECES for piece in resolutions.values()):
        raise ChessSolverError("adjudication contains an invalid piece")
    occupancies = [item.occupancy() for item in transcriptions]
    pieces: dict[str, str | None] = {}
    for square in _SQUARES:
        values = {occupancy[square] for occupancy in occupancies}
        pieces[square] = resolutions[square] if len(values) > 1 else next(iter(values))
    placement = _fen_placement(pieces)
    fen = f"{placement} {side_to_move} {castling} {en_passant} 0 1"
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ChessSolverError("reconciled FEN is syntactically invalid") from exc
    if not board.is_valid():
        raise ChessSolverError(f"reconciled FEN is illegal (status={int(board.status())})")
    return ValidatedChessPosition(
        fen=fen,
        side_to_move=side_to_move,
        sensors=comparison.sensors,
        adjudicated_squares=comparison.disagreements,
    )


class StockfishEngine:
    """Local UCI adapter; no model may substitute for the engine result."""

    def __init__(self, executable: Path) -> None:
        path = executable.resolve()
        if not path.is_file():
            raise ChessSolverError("configured Stockfish executable does not exist")
        self._executable = path

    def analyse(
        self,
        position: ValidatedChessPosition,
        *,
        depth: int,
        multipv: int,
    ) -> tuple[EngineLine, ...]:
        board = chess.Board(position.fen)
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(self._executable))
            try:
                raw = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=multipv,
                )
            finally:
                engine.quit()
        except Exception as exc:
            raise ChessSolverError("Stockfish analysis failed") from exc
        infos = raw
        lines: list[EngineLine] = []
        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            actual_depth = info.get("depth")
            if not pv or score is None or not isinstance(actual_depth, int):
                raise ChessSolverError("Stockfish returned an incomplete principal variation")
            move = pv[0]
            pov = score.pov(board.turn)
            mate = pov.mate()
            cp = pov.score()
            lines.append(
                EngineLine(
                    move_uci=move.uci(),
                    depth=actual_depth,
                    score_cp=cp,
                    mate_in=mate,
                )
            )
        return tuple(lines)


def solve_engine_position(
    position: ValidatedChessPosition,
    engine: ChessEnginePort,
    *,
    depth: int = 18,
    multipv: int = 3,
    minimum_winning_cp: int = 150,
) -> ChessSolution:
    if depth < 8 or multipv < 2:
        raise ValueError("chess engine verification requires depth >= 8 and multipv >= 2")
    lines = tuple(engine.analyse(position, depth=depth, multipv=multipv))
    if len(lines) < 2 or len(lines) > multipv:
        raise ChessSolverError("engine must return bounded alternative lines")
    board = chess.Board(position.fen)
    for line in lines:
        move = chess.Move.from_uci(line.move_uci)
        if move not in board.legal_moves:
            raise ChessSolverError("engine returned an illegal move")
    best = lines[0]
    winning = (best.mate_in is not None and best.mate_in > 0) or (
        best.score_cp is not None and best.score_cp >= minimum_winning_cp
    )
    if not winning:
        raise ChessSolverError("engine best move is not verified as winning")
    answer = legal_san(position, best.move_uci)
    return ChessSolution(
        position=position,
        answer=answer,
        engine_lines=lines,
        winning_verified=True,
    )


def legal_san(position: ValidatedChessPosition, move_uci: str) -> ChessMoveAnswer:
    board = chess.Board(position.fen)
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        raise ChessSolverError("engine move is not valid UCI") from exc
    if move not in board.legal_moves:
        raise ChessSolverError("cannot serialize illegal chess move")
    san = board.san(move)
    if board.parse_san(san) != move:
        raise ChessSolverError("SAN round-trip did not preserve the engine move")
    return ChessMoveAnswer(uci=move.uci(), san=san)


def _visual_square(row: int, column: int, orientation: BoardOrientation) -> str:
    if orientation is BoardOrientation.WHITE_BOTTOM:
        return f"{chr(ord('a') + column)}{8 - row}"
    return f"{chr(ord('h') - column)}{row + 1}"


def _fen_placement(pieces: Mapping[str, str | None]) -> str:
    ranks: list[str] = []
    for rank in range(8, 0, -1):
        cells: list[str] = []
        empty = 0
        for file_name in "abcdefgh":
            piece = pieces[f"{file_name}{rank}"]
            if piece is None:
                empty += 1
            else:
                if empty:
                    cells.append(str(empty))
                    empty = 0
                cells.append(piece)
        if empty:
            cells.append(str(empty))
        ranks.append("".join(cells))
    return "/".join(ranks)


__all__ = [
    "BoardBounds",
    "BoardOrientation",
    "BoardTranscription",
    "ChessEnginePort",
    "ChessSolution",
    "ChessSolverError",
    "ChessSquareImage",
    "EngineLine",
    "SquareObservation",
    "StockfishEngine",
    "TranscriptionComparison",
    "ValidatedChessPosition",
    "compare_transcriptions",
    "legal_san",
    "reconcile_chess_position",
    "solve_engine_position",
    "split_chessboard_image",
    "transcription_from_piece_map",
]
