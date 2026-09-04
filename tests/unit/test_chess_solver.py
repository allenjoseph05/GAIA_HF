from __future__ import annotations

from io import BytesIO
from pathlib import Path

import chess
import pytest
from PIL import Image

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.solvers import (
    BoardBounds,
    BoardOrientation,
    ChessSolverError,
    EngineLine,
    compare_transcriptions,
    legal_san,
    reconcile_chess_position,
    solve_engine_position,
    split_chessboard_image,
    transcription_from_piece_map,
)

TACTIC_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


class FakeEngine:
    def analyse(self, position: object, *, depth: int, multipv: int) -> tuple[EngineLine, ...]:
        assert depth == 18
        assert multipv == 3
        return (
            EngineLine(move_uci="h5f7", depth=18, mate_in=1),
            EngineLine(move_uci="h5g5", depth=18, score_cp=80),
            EngineLine(move_uci="h5h3", depth=18, score_cp=30),
        )


def piece_map(fen: str) -> dict[str, str]:
    board = chess.Board(fen)
    return {
        chess.square_name(square): piece.symbol()
        for square, piece in board.piece_map().items()
    }


def position_from_fen(fen: str = TACTIC_FEN):
    board = chess.Board(fen)
    first = transcription_from_piece_map("sensor_a", piece_map(fen))
    second = transcription_from_piece_map("sensor_b", piece_map(fen))
    return reconcile_chess_position(
        (first, second),
        side_to_move="w" if board.turn else "b",
        castling=board.castling_xfen(),
        en_passant=chess.square_name(board.ep_square) if board.ep_square is not None else "-",
    )


def synthetic_board_png() -> bytes:
    image = Image.new("RGB", (80, 80))
    pixels = image.load()
    assert pixels is not None
    for row in range(8):
        for column in range(8):
            color = (row * 25, column * 25, 100)
            for y in range(row * 10, (row + 1) * 10):
                for x in range(column * 10, (column + 1) * 10):
                    pixels[x, y] = color
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_square_mapping_preserves_white_and_black_orientations(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put_bytes(
        synthetic_board_png(),
        original_name="board.png",
        media_type="image/png",
        source=ArtifactSource.FIXTURE,
    )
    bounds = BoardBounds(left=0, top=0, right=80, bottom=80)
    white = split_chessboard_image(
        store,
        artifact,
        bounds=bounds,
        orientation=BoardOrientation.WHITE_BOTTOM,
    )
    black = split_chessboard_image(
        store,
        artifact,
        bounds=bounds,
        orientation=BoardOrientation.BLACK_BOTTOM,
    )

    assert white[0].square == "a8"
    assert white[-1].square == "h1"
    assert black[0].square == "h1"
    assert black[-1].square == "a8"
    with Image.open(BytesIO(store.read_bytes(store.get(white[0].artifact_id)))) as crop:
        assert crop.getpixel((5, 5)) == (0, 0, 100)


def test_dual_transcriptions_report_disagreements_and_require_adjudication() -> None:
    pieces = piece_map(TACTIC_FEN)
    altered = dict(pieces)
    altered.pop("h5")
    altered["g5"] = "Q"
    first = transcription_from_piece_map("sensor_a", pieces)
    second = transcription_from_piece_map("sensor_b", altered)
    comparison = compare_transcriptions((first, second))

    assert comparison.disagreements == ("g5", "h5")
    with pytest.raises(ChessSolverError, match="every and only"):
        reconcile_chess_position((first, second), side_to_move="w")
    reconciled = reconcile_chess_position(
        (first, second),
        side_to_move="w",
        adjudicated={"g5": None, "h5": "Q"},
        castling="KQkq",
    )
    assert reconciled.fen == TACTIC_FEN.replace(" 4 4", " 0 1")
    assert reconciled.adjudicated_squares == ("g5", "h5")


def test_illegal_fen_cannot_reach_engine() -> None:
    pieces = piece_map(TACTIC_FEN)
    pieces.pop("e8")
    first = transcription_from_piece_map("sensor_a", pieces)
    second = transcription_from_piece_map("sensor_b", pieces)
    with pytest.raises(ChessSolverError, match="illegal"):
        reconcile_chess_position((first, second), side_to_move="w")


def test_engine_move_is_legal_winning_and_serialized_by_board() -> None:
    position = position_from_fen()
    solution = solve_engine_position(position, FakeEngine())

    assert solution.winning_verified is True
    assert solution.answer.uci == "h5f7"
    assert solution.answer.san == "Qxf7#"
    board = chess.Board(position.fen)
    assert board.parse_san(solution.answer.san).uci() == solution.answer.uci


def test_san_rejects_illegal_move_and_engine_rejects_nonwinning_best() -> None:
    position = position_from_fen()
    with pytest.raises(ChessSolverError, match="illegal"):
        legal_san(position, "a1a8")

    class WeakEngine:
        def analyse(
            self,
            position: object,
            *,
            depth: int,
            multipv: int,
        ) -> tuple[EngineLine, ...]:
            return (
                EngineLine(move_uci="h5g5", depth=depth, score_cp=50),
                EngineLine(move_uci="h5h3", depth=depth, score_cp=40),
            )

    with pytest.raises(ChessSolverError, match="not verified as winning"):
        solve_engine_position(position, WeakEngine())


def test_orientation_disagreement_and_duplicate_sensor_identity_fail() -> None:
    pieces = piece_map(TACTIC_FEN)
    first = transcription_from_piece_map("same", pieces)
    duplicate = transcription_from_piece_map("same", pieces)
    with pytest.raises(ChessSolverError, match="independent"):
        compare_transcriptions((first, duplicate))
    other_orientation = transcription_from_piece_map(
        "other",
        pieces,
        orientation=BoardOrientation.BLACK_BOTTOM,
    )
    with pytest.raises(ChessSolverError, match="orientation"):
        compare_transcriptions((first, other_orientation))
