from __future__ import annotations

from pathlib import Path

from app import application_status, run_public_demo


def test_public_space_status_is_answer_free_and_submission_disabled() -> None:
    status = application_status()
    encoded = repr(status).casefold()

    assert status["submission"] == "disabled"
    assert "20/20" in status["status"]
    assert "candidate_answer" not in encoded
    assert "submitted_answer" not in encoded


def test_docker_context_excludes_private_runtime_and_secrets() -> None:
    patterns = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {"runs", ".runtime", ".env", ".env.*", "artifacts"} <= patterns


def test_public_demo_uses_only_synthetic_tasks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    report = run_public_demo()

    assert report["answers_are_synthetic"] is True
    assert report["submission_capability_present"] is False
    assert set(report["selected_task_ids"]) == {
        "demo-transformed-text",
        "demo-operation-table",
    }
