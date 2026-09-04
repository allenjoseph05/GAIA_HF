from __future__ import annotations

from importlib.metadata import version

from gaia_max import __version__
from gaia_max.cli import main


def test_runtime_version_matches_package_metadata() -> None:
    assert __version__ == version("gaia-max")


def test_about_command(capsys) -> None:
    exit_code = main(["about"])

    assert exit_code == 0
    assert "portfolio hardening complete" in capsys.readouterr().out


def test_config_command_is_redacted_and_safe(capsys, monkeypatch) -> None:
    secret = "hf_do_not_print_me"
    monkeypatch.setenv("HF_TOKEN", secret)

    exit_code = main(["config"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"submission_enabled": false' in output
    assert '"zero_cost_mode": true' in output
    assert secret not in output


def test_cli_safety_flags_override_submission_enabled_environment(
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("ALLOW_SUBMIT", "true")

    exit_code = main(["--dry-run", "--no-submit", "config"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"dry_run": true' in output
    assert '"allow_submit": false' in output
    assert '"submission_enabled": false' in output
