"""Safe operator CLI for public demos and private evaluation preflight."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from gaia_max import __version__
from gaia_max.clients import GaiaApiClient
from gaia_max.config import Settings
from gaia_max.current_run import (
    build_current_dry_run,
    preflight_current_candidate,
    prepare_current_attachments,
)
from gaia_max.portfolio_runtime import run_portfolio_demo
from gaia_max.profiles import load_profile_registry, reconcile_profile_registry
from gaia_max.snapshot import (
    StoredQuestionSnapshot,
    build_question_snapshot,
    persist_question_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaia",
        description="Build, inspect, and run the GAIA maximum-score agent.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run safety settings for this command.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Force submission off for this command.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("about", help="Show the current implementation stage.")
    subparsers.add_parser("config", help="Show a redacted configuration and capability report.")
    subparsers.add_parser(
        "questions",
        help="Read the current official question inventory without showing answers or submitting.",
    )
    subparsers.add_parser(
        "snapshot",
        help="Hash and persist the current public question set without submitting.",
    )
    subparsers.add_parser(
        "profiles",
        help="Compare the answer-free task profiles with the latest local snapshot.",
    )
    subparsers.add_parser(
        "prepare-run",
        help="Acquire and validate current attachments without solving or submitting.",
    )
    subparsers.add_parser(
        "dry-run-current",
        help="Validate every private current candidate and emit only answer-redacted status.",
    )
    subparsers.add_parser(
        "preflight-current",
        help="Freeze the private current candidate only if all 20 tasks are ready.",
    )
    solve = subparsers.add_parser(
        "solve",
        help="Solve one bundled synthetic task through the complete graph stack.",
    )
    solve.add_argument(
        "task_id",
        choices=("demo-transformed-text", "demo-operation-table"),
    )
    solve.add_argument("--run-id", default="portfolio-solve")
    run = subparsers.add_parser(
        "run",
        help="Run the complete, public synthetic portfolio scenario.",
    )
    run.add_argument("--run-id", default="portfolio-demo")
    resume = subparsers.add_parser(
        "resume",
        help="Resume or safely reuse checkpoints for a public synthetic run.",
    )
    resume.add_argument("--run-id", default="portfolio-demo")
    return parser


async def _show_question_inventory(settings: Settings) -> None:
    async with GaiaApiClient(settings) as client:
        questions = await client.get_questions()

    task_ids = [question.task_id for question in questions]
    summary = {
        "question_count": len(questions),
        "unique_task_ids": len(set(task_ids)),
        "attachment_count": sum(question.file_name is not None for question in questions),
        "submission_capability_present": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


async def _create_question_snapshot(settings: Settings) -> int:
    async with GaiaApiClient(settings) as client:
        questions = await client.get_questions()

    source_url = f"{str(settings.gaia_api_url).rstrip('/')}/questions"
    snapshot = build_question_snapshot(
        questions,
        source_url=source_url,
        task_profiles_version=settings.task_profiles_version,
    )
    path = persist_question_snapshot(
        snapshot,
        questions,
        snapshots_dir=settings.runs_dir / "snapshots",
    )
    expected_count_matches = snapshot.count == settings.expected_question_count
    summary = {
        "attachment_count": len(snapshot.attachment_task_ids),
        "expected_count": settings.expected_question_count,
        "expected_count_matches": expected_count_matches,
        "question_count": snapshot.count,
        "snapshot_path": str(path),
        "snapshot_sha256": snapshot.snapshot_sha256,
        "task_profiles_version": snapshot.task_profiles_version,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if expected_count_matches else 2


def _inspect_profiles(settings: Settings) -> int:
    snapshot_paths = list((settings.runs_dir / "snapshots").glob("questions-*.json"))
    if not snapshot_paths:
        print(json.dumps({"error": "no local question snapshot; run `gaia snapshot` first"}))
        return 2

    latest_path = max(snapshot_paths, key=lambda path: path.stat().st_mtime_ns)
    stored = StoredQuestionSnapshot.model_validate_json(latest_path.read_text(encoding="utf-8"))
    registry = load_profile_registry(settings.task_profiles_path)
    audit = reconcile_profile_registry(registry, stored.snapshot)
    summary = {
        "matched_profiles": len(audit.matched_task_ids),
        "missing_profile_task_ids": audit.missing_profile_task_ids,
        "profile_version_matches": audit.profile_version_matches,
        "snapshot_hash_matches": audit.snapshot_hash_matches,
        "stale_profile_task_ids": audit.stale_profile_task_ids,
        "submission_safe": audit.submission_safe,
        "unknown_profile_task_ids": audit.unknown_profile_task_ids,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if audit.submission_safe else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings_overrides: dict[str, bool] = {}
    if args.dry_run:
        settings_overrides.update(dry_run=True, allow_submit=False)
    if args.no_submit:
        settings_overrides["allow_submit"] = False

    if args.command == "about":
        print(f"GAIA Maximum-Score Agent {__version__} (portfolio hardening complete)")
        return 0

    if args.command == "config":
        settings = Settings.model_validate(settings_overrides)
        print(json.dumps(settings.public_summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "questions":
        asyncio.run(_show_question_inventory(Settings.model_validate(settings_overrides)))
        return 0
    if args.command == "snapshot":
        return asyncio.run(_create_question_snapshot(Settings.model_validate(settings_overrides)))
    if args.command == "profiles":
        return _inspect_profiles(Settings.model_validate(settings_overrides))
    if args.command == "prepare-run":
        report = asyncio.run(
            prepare_current_attachments(Settings.model_validate(settings_overrides))
        )
        print(report.model_dump_json(indent=2))
        return 0 if report.ready_attachments == report.expected_attachments else 2
    if args.command == "dry-run-current":
        settings_overrides.update(dry_run=True, allow_submit=False)
        report = build_current_dry_run(Settings.model_validate(settings_overrides))
        print(report.model_dump_json(indent=2))
        return 0 if report.ready_count == report.task_count else 2
    if args.command == "preflight-current":
        settings_overrides.update(dry_run=True, allow_submit=False)
        report = preflight_current_candidate(Settings.model_validate(settings_overrides))
        print(report.model_dump_json(indent=2))
        return 0 if report.valid else 2
    if args.command in {"solve", "run", "resume"}:
        selected = (args.task_id,) if args.command == "solve" else ()
        report = asyncio.run(
            run_portfolio_demo(
                Settings.model_validate(settings_overrides),
                run_id=args.run_id,
                selected_task_ids=selected,
            )
        )
        print(report.model_dump_json(indent=2))
        return 0 if all(task.status.value == "ready" for task in report.tasks) else 2

    parser.print_help()
    return 0
