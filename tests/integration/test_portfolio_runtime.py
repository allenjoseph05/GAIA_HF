from __future__ import annotations

from pathlib import Path

import pytest

from gaia_max.config import Settings
from gaia_max.domain import TaskStatus
from gaia_max.portfolio_runtime import run_portfolio_demo
from gaia_max.run_graph import RunStatus, TaskDispatchAction


@pytest.mark.asyncio
async def test_public_demo_executes_real_graph_and_reuses_checkpoints(tmp_path: Path) -> None:
    settings = Settings.model_validate(
        {
            "runs_dir": tmp_path / "runs",
            "artifacts_dir": tmp_path / "artifacts",
            "max_task_concurrency": 2,
        }
    )

    first = await run_portfolio_demo(settings, run_id="integration-demo")
    second = await run_portfolio_demo(settings, run_id="integration-demo")

    assert first.status is RunStatus.COMPLETE
    assert first.answers_are_synthetic is True
    assert first.submission_capability_present is False
    assert [task.status for task in first.tasks] == [TaskStatus.READY, TaskStatus.READY]
    assert [task.checkpoint_action for task in first.tasks] == [
        TaskDispatchAction.START,
        TaskDispatchAction.START,
    ]
    assert [task.checkpoint_action for task in second.tasks] == [
        TaskDispatchAction.REUSE,
        TaskDispatchAction.REUSE,
    ]
    assert {task.task_id: task.serialized_answer for task in first.tasks} == {
        "demo-transformed-text": "south",
        "demo-operation-table": "a, z",
    }


@pytest.mark.asyncio
async def test_public_demo_can_solve_exactly_one_selected_task(tmp_path: Path) -> None:
    settings = Settings.model_validate({"runs_dir": tmp_path / "runs"})

    report = await run_portfolio_demo(
        settings,
        run_id="one-task",
        selected_task_ids=("demo-operation-table",),
    )

    assert report.selected_task_ids == ("demo-operation-table",)
    assert len(report.tasks) == 1
    assert report.tasks[0].serialized_answer == "a, z"
