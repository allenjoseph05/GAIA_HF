# GitHub Publication Plan

Status: published and verified.

## Publication principles

- Treat the current tree as a recovered, verified release snapshot because the
  original Git metadata is unavailable.
- Do not manufacture historical feature commits from already-completed work.
- Keep one concern per commit and one reviewed push per branch. A push is a
  network operation on a branch, not a useful unit per file.
- Never commit `.env`, tokens, benchmark runtime state, caches, databases,
  generated packages, or local virtual environments.
- Create the GitHub repository empty: do not add a README, license, or
  `.gitignore` on GitHub.

## Branch and commit ledger

| Order | Branch | Commit | File scope | Gate before push | Push count |
| --- | --- | --- | --- | --- | --- |
| 1 | `main` | `chore(release): publish recovered GAIA Max 1.0.0 snapshot` | All reviewed source, tests, configuration, CI, documentation, lockfile, and repository policy files | secret scan; ignored-file audit; Ruff; Pyright; full pytest; package build | 1 |
| 2 | `chore/github-publication` | `chore(repo): update GitHub publication controls` | Only publication-specific metadata or documentation changed after the snapshot | Markdown/path review; `git diff --check` | 1 |
| 3 | `ci/supply-chain` | `ci: harden dependency and workflow controls` | Only `.github/**` plus dependency-policy files | workflow review; frozen install; full CI locally where reproducible | 1 |
| 4 | `feat/agent-orchestration` | Conventional commits scoped to LangGraph routing, state, checkpointing, and tool policy | `task_graph.py`, orchestration collaborators, and directly corresponding tests | focused tests, then full suite | 1 per completed branch |
| 5 | `feat/research-intelligence` | Conventional commits scoped to LlamaIndex retrieval, smolagents delegation, evidence composition, and telemetry | `research/**`, `observability.py`, direct configuration, and directly corresponding tests | focused tests, type check, then full suite | 1 per completed branch |
| 6 | `feat/solver-capabilities` | Conventional commits scoped to deterministic solvers, retrieval, media, or verification | One capability family and directly corresponding tests per branch | focused tests, then full suite | 1 per completed branch |
| 7 | `docs/architecture` | `docs: update architecture and portfolio evidence` | Documentation and verified result artifacts only | factual review; link check; `git diff --check` | 1 |

Only order 1 is needed for the initial publication. Orders 2-7 are the branch
taxonomy for genuine follow-up work; empty or artificial branches must not be
created.

## Initial snapshot allowlist

The first commit may contain only these paths:

```text
.dockerignore
.env.example
.gitattributes
.github/
.gitignore
app.py
config/
docs/
Dockerfile
GAIA_MAX_SCORE_ENGINEERING_SPEC.md
LICENSE
pyproject.toml
README.md
SECURITY.md
src/
tests/
uv.lock
```

The following must remain absent from every commit:

```text
.env
.hypothesis/
.package-test/
.pytest_cache/
.ruff_cache/
.runtime/
.uv-cache/
.venv/
__pycache__/
artifacts/
build/
dist/
runs/
*.egg-info/
*.py[cod]
*.sqlite*
*.db*
```

## Local preparation sequence

Run from the project root after the publication audit succeeds:

```powershell
git init -b main
git add -- .dockerignore .env.example .gitattributes .github .gitignore app.py config docs Dockerfile GAIA_MAX_SCORE_ENGINEERING_SPEC.md LICENSE pyproject.toml README.md SECURITY.md src tests uv.lock
git status --short
git diff --cached --check
git commit -m "chore(release): publish recovered GAIA Max 1.0.0 snapshot"
git tag -a v1.0.0 -m "GAIA Max 1.0.0 recovered and verified release"
```

The explicit allowlist is intentional: `git add .` is not part of the release
procedure.

## Remote publication sequence

Replace `<REMOTE_URL>` with the URL of the empty repository:

```powershell
git remote add origin <REMOTE_URL>
git remote -v
git push -u origin main
git push origin v1.0.0
```

For every later category branch:

```powershell
git switch main
git pull --ff-only
git switch -c <CATEGORY_BRANCH>
# Make one coherent change with its tests and documentation.
git add -- <EXPLICIT_FILES>
git diff --cached --check
git commit -m "<CONVENTIONAL_COMMIT>"
git push -u origin <CATEGORY_BRANCH>
```

Open one pull request into `main`, require the quality workflow to pass, squash
only if the branch contains fix-up commits, then delete the remote branch.

## GitHub repository controls

After `main` exists remotely:

1. Set `main` as the default branch.
2. Protect `main`: require a pull request, one approval when collaborators are
   present, required status checks, resolved conversations, and no force pushes.
3. Enable secret scanning, push protection, dependency graph, Dependabot alerts,
   and private vulnerability reporting when the repository plan supports them.
4. Add repository topics such as `ai-agents`, `langgraph`, `llamaindex`,
   `smolagents`, `rag`, `observability`, and `huggingface`.
5. Keep the verified result phrasing precise: Hugging Face Agents Course Unit 4
   GAIA assignment, 20/20 questions, 100/100 score; do not claim completion of
   the full GAIA benchmark.

## Release acceptance record

Record the exact commands and results immediately before the first push:

```text
Secret scan:        PASS - no credential-shaped values in the allowlist
Ignored-file audit: PASS - local secrets, state, caches, builds, and bytecode excluded
Ruff:               PASS - all checks passed
Pyright:            PASS - 0 errors, 0 warnings, 0 informations
Pytest:             PASS - 674 passed, 1 third-party Python 3.16 deprecation warning
Package build:      PASS - sdist and universal wheel built for 1.0.0
Runtime smoke test: PASS - synthetic LangGraph run completed; submission unavailable
Remote URL:         https://github.com/allenjoseph05/GAIA_HF
main push:          PASS - bdf1115 published as a fast-forward
v1.0.0 tag push:    PASS - annotated release tag published
```
