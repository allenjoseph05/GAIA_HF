"""Allow ``python -m gaia_max`` to invoke the CLI."""

from gaia_max.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
