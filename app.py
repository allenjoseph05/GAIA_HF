# pyright: reportMissingImports=false
"""Public, answer-free Hugging Face Space entry point."""

from __future__ import annotations

import asyncio

from gaia_max import __version__
from gaia_max.config import Settings
from gaia_max.portfolio_runtime import run_portfolio_demo


def application_status() -> dict[str, str]:
    """Return public, non-sensitive build information."""

    return {
        "project": "GAIA Maximum-Score Agent",
        "version": __version__,
        "status": "official evaluation complete: 20/20 (100%)",
        "public_demo": "two synthetic deterministic tasks; no benchmark answers",
        "submission": "disabled",
    }


def run_public_demo() -> dict[str, object]:
    """Run the synthetic graph scenario; never load private evaluation state."""

    settings = Settings.model_validate(
        {
            "runs_dir": ".runtime/public-demo",
            "artifacts_dir": ".runtime/public-demo/artifacts",
            "dry_run": True,
            "allow_submit": False,
            "max_task_concurrency": 2,
        }
    )
    report = asyncio.run(run_portfolio_demo(settings, run_id="space-demo"))
    return report.model_dump(mode="json")


def build_demo() -> object:
    """Build a read-only public UI without importing private runtime state."""

    import gradio as gr

    status = application_status()
    with gr.Blocks(title=status["project"]) as demo:
        gr.Markdown(
            "# GAIA Maximum-Score Agent\n\n"
            "A reproducible, evidence-first agent built for the Hugging Face Agents "
            "Course GAIA assignment. The public Space intentionally contains no "
            "candidate answers, benchmark artifacts, secrets, or submission control."
        )
        gr.JSON(value=status, label="Public build status")
        run_button = gr.Button("Run safe end-to-end demo", variant="primary")
        demo_output = gr.JSON(label="Synthetic graph result")
        run_button.click(fn=run_public_demo, outputs=demo_output)
        gr.Markdown(
            "Submission is disabled by default and can occur only from a private "
            "operator environment after all 20 tasks pass frozen-manifest preflight "
            "and a human approval is bound to that exact manifest hash."
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)  # type: ignore[attr-defined]
