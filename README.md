---
title: GAIA Maximum-Score Agent
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
---

# GAIA Maximum-Score Agent

This repository is a learning project and an engineering project at the same time.

The learning goal is to understand the agent concepts and frameworks taught in the free Hugging Face Agents Course. The engineering goal is to build a legitimate, reproducible agent for the Unit 4 GAIA final assignment that is designed for the maximum current score: 20 correct answers out of 20.

The original source of truth is [GAIA_MAX_SCORE_ENGINEERING_SPEC.md](GAIA_MAX_SCORE_ENGINEERING_SPEC.md). The documents below translate that large specification into beginner-friendly diagrams, implementation decisions, lessons, and ordered stories that we can complete one at a time.

## Start here

Read these documents in order:

1. [Beginner architecture](docs/01_BEGINNER_ARCHITECTURE.md) — what we are building and why, explained visually.
2. [Detailed system design](docs/02_SYSTEM_DESIGN.md) — components, state, graphs, solvers, verification, storage, and deployment.
3. [Learning and collaboration guide](docs/03_LEARNING_WORKFLOW.md) — how each coding story will also be a lesson.
4. [Implementation backlog](docs/04_IMPLEMENTATION_BACKLOG.md) — the ordered epics and stories we will implement.
5. [Requirements traceability](docs/05_REQUIREMENTS_TRACEABILITY.md) — where every section of the original specification is handled.
6. [Free execution and official scoring](docs/06_FREE_EXECUTION_AND_SCORING.md) — what Hugging Face scores, what it does not run, and how the zero-spend path works.

7. [Portfolio defense guide](docs/07_PORTFOLIO_DEFENSE.md) — the honest claims, major decisions, trust boundaries, and limitations.
8. [Recovery checkpoint](docs/08_RECOVERY_CHECKPOINT.md) — the latest post-shutdown validation state and remaining external gate.

## The central design rule

> Use LangGraph to control the workflow. Use an LLM to research, interpret, and reason where necessary. Use deterministic software to download, parse, calculate, execute, sort, verify, and format whenever possible.

This means we are not building one large chatbot that improvises everything. We are building a controlled system of specialist solvers.

## Shipped technology and boundaries

| Technology | Role in this project |
|---|---|
| LangGraph | Typed task/run orchestration, routing, checkpoints, resume, and isolated submission approval |
| smolagents | Bounded `ToolCallingAgent` used only by the deep-research fallback; strict JSON output and step limits |
| LlamaIndex | Safety-approved long-document chunking, BM25 retrieval, and optional dense reciprocal-rank fusion |
| Pydantic | Strict schemas at question, profile, tool, evidence, answer, and report boundaries |
| Deterministic specialists | Exact transforms, tables, Python, spreadsheets, historical extraction, media, and chess components |
| OpenTelemetry / Langfuse | Content-free graph, research, retrieval, and tool spans; Langfuse v4 is an optional sink |
| Gradio | Read-only public status and a synthetic end-to-end graph demonstration |
| Hugging Face Hub | Space deployment, authentication, model access, and official gated attachment fallback |

The default deterministic runtime remains lightweight. Research and observability are
installable extras and activate only through explicit dependency injection. The fallback
agent cannot submit, execute code, expand its tool allowlist, cite unseen passages, or
bypass the existing evidence verifier and serializer.

## Course and benchmark boundaries

- The project uses only the public evaluation questions and legitimate public evidence.
- The official gated GAIA dataset may be used to retrieve attachments when the course file endpoint fails.
- The target `Final answer` field must never be read, logged, indexed, or committed.
- We do not search for leaked answers, repositories containing target answers, or `<task_id> answer`.
- The public Space contains the implementation, not cached benchmark artifacts, secrets, private traces, or candidate answers.
- Official submission is disabled by default and requires a passing 20-task preflight plus explicit human approval.

## Current implementation status

- [x] Engineering specification reviewed
- [x] Official course and current scoring contract reviewed
- [x] Architecture documented
- [x] Ordered learning backlog documented
- [x] Milestones A-E and the score-critical F/H controls implemented
- [x] GAIA-A01 project skeleton
- [x] GAIA-A02 safe configuration and zero-cost guard
- [x] Initial GAIA-A03 core domain models
- [x] GAIA-A04 read-only official course API client
- [x] GAIA-A05 immutable live question snapshots
- [x] GAIA-A06 answer-free task-profile registry
- [x] GAIA-A07 content-addressed artifact storage
- [x] GAIA-A08 defensive attachment validation
- [x] GAIA-A09 official endpoint attachment retrieval
- [x] GAIA-A10 official gated-dataset attachment fallback
- [x] GAIA-A11 deterministic output-contract parsing
- [x] GAIA-A12 structured model-assisted task analysis
- [x] GAIA-A13 typed semantic answers and deterministic serializer
- [x] GAIA-A14 structured final-answer validation
- [x] GAIA-A15 minimal typed LangGraph TaskGraph
- [x] GAIA-A16 async SQLite checkpointing, history, and resume
- [x] GAIA-A17 deterministic solver routing and explicit specialist registry
- [x] GAIA-A18 LangGraph RunGraph, bounded task fan-out, and checkpoint-aware recovery
- [x] GAIA-A19 answer-redacted dry-run report and structural preflight
- [x] GAIA-C09 guarded LlamaIndex long-document retrieval and hybrid fusion
- [x] GAIA-F01-F02 bounded smolagents research worker and evidence-gap replanning
- [x] GAIA-B01 deterministic transformed-text solver
- [x] GAIA-B02 operation-table parser and exhaustive solver
- [x] GAIA-B03 structured sports/table reduction
- [x] GAIA-B04 constrained Python inspection and execution
- [x] GAIA-B05 answer-free workbook structural inspection
- [x] GAIA-B06 independent spreadsheet Decimal calculations
- [x] GAIA-B07 deterministic replay verification policies
- [x] GAIA-B08 property and poisoned-implementation tests
- [x] Milestone B deterministic-solvers gate
- [x] GAIA-C01 typed search and page-retrieval provider boundaries
- [x] GAIA-C02 independent primary/secondary search orchestration
- [x] GAIA-C03 bounded HTTP extraction and page-retrieval fallback ladder
- [x] GAIA-C04 prompt-injection quarantine and model/source boundaries
- [x] GAIA-C05 MediaWiki historical revision client and exact old-revision provenance
- [x] GAIA-C06 deterministic dated discography and FAC-nominator extraction
- [x] GAIA-C07 bounded Wayback capture discovery and temporal-validity policy
- [x] GAIA-C08 page-addressable PDF extraction, render/OCR fallback, and exact context
- [x] GAIA-C10 exact scholarly funding and specimen-deposition relationships
- [x] GAIA-C11 stable cross-lingual identity and source-backed Romanization
- [x] GAIA-C12 structured HTML-table normalization and deterministic reduction
- [x] GAIA-C13 claim-level evidence readiness and snippet exclusion
- [x] GAIA-D01-D11 reproducible media, timestamped ASR, captions, exact speech, frames, and independent species analysis
- [x] GAIA-E01-E05 chess image mapping, dual transcription, legal FEN, Stockfish, and SAN
- [x] GAIA-F04-F09 structural, source, independent, adjudication, and risk verification
- [x] GAIA-F10 answer-safe OpenTelemetry with optional Langfuse v4 export
- [x] Current 20-task answer-redacted audit: 20 READY and 0 BLOCKED
- [x] Private candidate preflight freezes exactly 20 unique, serialization-valid tasks
- [x] Docker/Gradio public Space shell contains no private runtime data or submission control
- [x] One-shot submission graph is disabled by default and requires hash-bound approval
- [x] Current 20-task dry run completed
- [x] Public Hugging Face Space deployed
- [x] Official submission completed: 20/20 (100%)
- [x] Public synthetic end-to-end TaskGraph/RunGraph demonstration
- [x] Strict production Pyright gate: 0 errors
- [x] Locked dependencies, CI, security model, and Apache-2.0 license

The score-critical run is complete. When the course `/files` endpoint returned no
artifact for five tasks, the official gated-dataset fallback retrieved and validated only
their requested attachment bytes without reading the answer column. All 20 tasks passed
answer-redacted evidence, verification, and output-contract checks. The exact candidate
was frozen, explicitly approved by hash, and scored 20/20. The sanitized result record is
[docs/results/official-score.json](docs/results/official-score.json); it contains no
questions, answers, gated content, or secrets.

Execution now follows the [maximum-score critical path](docs/04_IMPLEMENTATION_BACKLOG.md#maximum-score-execution-policy), not mechanical story-ID order. Direct coverage of the current 20 task shapes, verification that prevents exact-match losses, and the end-to-end dry run come before optional framework breadth, dashboards, and UI polish.

## Safe current-run commands

```bash
gaia --dry-run --no-submit dry-run-current
gaia --dry-run --no-submit preflight-current
```

Both commands emit answer-redacted reports. `preflight-current` writes a private frozen
manifest under the ignored `runs/` directory only if all expected task IDs are READY.
The public Docker Space is a read-only status surface; submission remains a separate
private operator action.

## Reproducible end-to-end demonstration

The official candidate is intentionally private, so the repository includes two
synthetic tasks that run through the real snapshot, profile, routing, solver,
verification, serialization, SQLite checkpoint, resume, and run-collection stack:

```bash
gaia --dry-run --no-submit solve demo-transformed-text --run-id local-solve
gaia --dry-run --no-submit run --run-id local-demo
gaia --dry-run --no-submit resume --run-id local-demo
```

`resume` reuses terminal checkpoints and resumes interrupted ones. Unsupported routes
remain blocked. Demo answers are deliberately synthetic and no submission capability is
present.

## Local setup and validation

Python 3.11 through 3.14 is supported. Install the locked development environment and
keep the default safety gates enabled:

```bash
python -m uv sync --frozen --extra dev
python -m uv run gaia --dry-run --no-submit config
```

An editable `python -m pip install -e ".[dev]"` setup is also supported.
Install the optional hybrid stack with:

```bash
python -m uv sync --frozen --extra dev --extra research --extra observability
```

`research-local` additionally enables local Hugging Face embeddings on Python versions
below 3.14; production can instead inject any LlamaIndex-compatible embedding model.

Run the public quality gates with:

```bash
python -m ruff check .
python -m pyright
python -m pytest -q
```

The quality gate covers lint, strict production typing, unit/property/integration tests,
the installed CLI, and the Docker image. Private runtime data belongs only
under `runs/` or `.runtime/`; both paths are excluded from Git and the Docker context.

## Defending the design

The interview-oriented rationale, execution path, trust boundaries, trade-offs, and
limitations are in [docs/07_PORTFOLIO_DEFENSE.md](docs/07_PORTFOLIO_DEFENSE.md). The
enforceable threat model is in [SECURITY.md](SECURITY.md). They explicitly separate
integrated behavior from reusable specialist components and future extensions.

## Docker Space

The public UI is submission-disabled and includes only a safe synthetic graph demo:

```bash
docker build -t gaia-max-space .
docker run --rm -p 7860:7860 gaia-max-space
```

The image installs only the dependencies needed by the public demonstration and launches
Gradio on port 7860. Heavy ASR, FFmpeg, and Stockfish capabilities stay in the private
runtime rather than expanding the public attack surface. The image does not copy local
candidates, attachments, secrets, traces, or test artifacts. Space deployment and
private official submission remain separate capabilities.

## Gated attachment access

If the official course file endpoint cannot serve an attachment, accept the official
GAIA dataset access conditions and provide a read-capable `HF_TOKEN` in the private
environment. The fallback downloads only the exact requested artifact and never reads
the gated `Final answer` column. Do not put the token or downloaded files in public
Space files.

## Official references

- Course: https://huggingface.co/learn/agents-course/en/unit0/introduction
- LangGraph unit: https://huggingface.co/learn/agents-course/unit2/langgraph/introduction
- Unit 4 assignment: https://huggingface.co/learn/agents-course/unit4/hands-on
- Certificate page: https://huggingface.co/learn/agents-course/unit4/get-your-certificate
- Current questions: https://agents-course-unit4-scoring.hf.space/questions
- Current scorer source: https://huggingface.co/spaces/agents-course/Unit4_scoring/blob/main/main.py
- Official GAIA dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
- smolagents Open Deep Research: https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research
