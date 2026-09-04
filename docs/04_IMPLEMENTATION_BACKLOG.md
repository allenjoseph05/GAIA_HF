# Ordered Implementation Backlog

## How to use this backlog

Story IDs remain the traceability and dependency order, but implementation follows the maximum-score critical path below. Every story includes a learning goal and an observable acceptance condition. A milestone is complete only when all of its stories and milestone gate are complete.

The order follows the source specification:

```text
A. Contract and infrastructure
B. Deterministic solvers
C. Web and historical research
D. Audio and video
E. Chess
F. Ensemble verification
G. Current evaluation dry run
H. Deployment and submission
```

## Maximum-score execution policy

The live evaluation has 20 equally weighted questions, so direct task coverage and an end-to-end dry run take priority over completing every backlog story. Checked boxes still mean implemented; priority does not change completion status.

| Priority | Stories | Execution rule |
|---|---|---|
| Score-critical now | C05-C08, C10-C12, minimal C13; D01-D11; E01-E05; minimal F01-F02 and F04-F06/F08; G01-G06; H01, H03-H04, H06 | Build the smallest tested capability that solves or verifies a current task shape. |
| Conditional | F07, H02, H05 | Implement only when a critical solver or the dry run proves it is needed. |
| Deferred until coverage | F03, F09-F12, H07, elaborate dashboards/observability/UI | Do not block the 20-task dry run or submission path on these stories. |

Within the critical path, work in this order: historical web; scholarly/cross-lingual/table research; audio and video; chess; minimum verification; full dry run and repairs; deployment and submission. Revisit conditional and deferred work only when evidence from the dry run justifies it.

## Story template

For detailed implementation work, expand a story using:

```text
Story ID and title
User story
Why it matters
Learning objectives
Scope
Out of scope
Design/interfaces
Acceptance criteria
Tests
Demonstration
Dependencies
Specification references
```

# Milestone A — Contract and infrastructure

Goal: create the safe, typed, checkpointed foundation on which every solver depends.

## Epic A1 — Project foundation

### [x] GAIA-A01 — Create the Python project skeleton

**User story:** As a learner and developer, I want a standard Python package layout so every later component has a predictable home.

**Learn:** `pyproject.toml`, `src/` layout, dependency groups, console entry points, and test discovery.

**Build:** Package skeleton, `pyproject.toml`, minimal CLI, test directories, `.gitignore`, and placeholder `app.py`.

**Acceptance:** The package installs locally; `gaia --help` runs; one smoke test passes; runtime data and `.env` are ignored.

**Dependencies:** None.

### [x] GAIA-A02 — Define configuration and safe defaults

**User story:** As the operator, I want validated configuration so missing providers fail clearly and official submission is impossible by default.

**Learn:** Environment variables, Pydantic settings, secrets, dependency injection, and fail-safe defaults.

**Build:** `Settings`, `.env.example`, capability report, `DRY_RUN=true`, and `ALLOW_SUBMIT=false` defaults.

**Acceptance:** Invalid configuration produces structured errors; secrets are redacted; tests prove submission defaults to disabled.

**Dependencies:** A01.

### [x] GAIA-A03 — Define core domain models

**User story:** As every graph node, I want shared validated types so components exchange structured data rather than ambiguous dictionaries.

**Learn:** Pydantic models, enums, discriminated unions, validation, and JSON serialization.

**Build:** Models for questions, snapshots, artifacts, profiles, temporal constraints, output contracts, evidence, candidates, verification, risk, and task results.

**Acceptance:** Valid fixtures round-trip through JSON; invalid states are rejected; no model includes a gold-answer field.

**Dependencies:** A01.

## Epic A2 — Official API and snapshot integrity

### [x] GAIA-A04 — Implement the read-only official API client

**User story:** As the RunGraph, I want a typed client for official questions and attachments so external API behavior is isolated.

**Learn:** Async `httpx`, typed adapters, timeouts, and error classification.

**Build:** `/questions`, `/random-question`, and `/files/{task_id}` read methods. Do not implement submission yet.

**Acceptance:** Mock-server integration tests cover success, malformed JSON, 404, 429, 5xx, and timeout behavior.

**Dependencies:** A02, A03.

### [x] GAIA-A05 — Create immutable question snapshots

**User story:** As the operator, I want every run bound to an exact public question set so stale profiles cannot silently submit.

**Learn:** Canonical JSON, SHA-256, stable ordering, and integrity checks.

**Build:** Fetch, normalize, hash, persist, and compare snapshots; calculate per-question hashes and attachment inventory.

**Acceptance:** Reordered API data produces the same canonical hash; question changes produce a different task and snapshot hash; duplicate task IDs are rejected.

**Dependencies:** A04.

### [x] GAIA-A06 — Implement the answer-free task-profile registry

**User story:** As the router, I want versioned public task profiles so current known task shapes route deterministically without storing answers.

**Learn:** Configuration-as-data, profile versioning, and benchmark hygiene.

**Build:** YAML/JSON registry containing task class, solver route, risk policy, and output-contract hints keyed by task ID and question hash.

**Acceptance:** The registry contains no candidate/gold values; a hash mismatch marks a profile stale and blocks submission.

**Dependencies:** A03, A05.

## Epic A3 — Attachment resolution and artifacts

### [x] GAIA-A07 — Build content-addressed artifact storage

**User story:** As a solver, I want immutable artifact references so repeated calculations and verification use identical bytes.

**Learn:** Content-addressed storage, atomic writes, metadata, and path safety.

**Build:** SHA-256 artifact store with metadata, safe filenames, atomic placement, and lookup.

**Acceptance:** Identical bytes deduplicate; altered bytes receive a different ID; unsafe filenames cannot escape the artifact root.

**Dependencies:** A03.

### [x] GAIA-A08 — Validate downloaded attachments

**User story:** As the attachment resolver, I want file validation so error pages and unsafe files never reach a solver.

**Learn:** MIME versus extension, magic bytes, size limits, and defensive file handling.

**Build:** Extension/MIME/magic validation for PNG/JPEG, MP3/audio, Python, and XLSX.

**Acceptance:** Synthetic valid fixtures pass; HTML error bodies, empty files, oversized files, and extension mismatches fail with reason codes.

**Dependencies:** A07.

### [x] GAIA-A09 — Implement official endpoint attachment retrieval

**User story:** As the system, I want the preferred official `/files` path with retries and validation.

**Learn:** Streaming downloads, retry boundaries, and immutable caching.

**Build:** Official API acquisition path connected to validators and artifact storage.

**Acceptance:** Successful bytes are cached with provenance; transient failures retry; deterministic 404 routes to fallback without looping.

**Dependencies:** A04, A07, A08.

### [x] GAIA-A10 — Implement official gated-dataset fallback

**User story:** As an attachment task, I want a legitimate fallback when the course file endpoint is broken.

**Learn:** Hugging Face dataset access, gated credentials, schema allowlists, and anti-leak design.

**Build:** Official GAIA attachment resolution using `HF_TOKEN`, while explicitly excluding the target `Final answer` field.

**Acceptance:** Fallback resolves official artifact bytes; tests prove the answer field is never requested, modeled, logged, or returned; missing access fails loudly.

**Dependencies:** A02, A07, A08, A09.

## Epic A4 — Output contracts

### [x] GAIA-A11 — Implement deterministic contract parsing rules

**User story:** As the TaskGraph, I want output requirements extracted before solving so the answer is produced in the required form.

**Learn:** Rule-based parsing, inclusions/exclusions, sorting, precision, and name scope.

**Build:** Rules for integers, decimals, currency, lists, first/surname/city/code, exact quote, and chess notation.

**Acceptance:** Public-question-shaped and synthetic fixtures produce expected contracts; ambiguous requirements are flagged rather than guessed.

**Dependencies:** A03.

### [x] GAIA-A12 — Add structured model-assisted task analysis

**User story:** As the analyzer, I want a model to propose ambiguous metadata while hard rules remain authoritative.

**Learn:** Structured model output, schema validation, and hybrid deterministic/LLM systems.

**Build:** Model adapter and reconciliation logic for task class, temporal constraints, filters, and risks.

**Acceptance:** Invalid model output cannot enter state; hard-signal disagreement is recorded; known profiles reconcile deterministically.

**Dependencies:** A06, A11.

### [x] GAIA-A13 — Implement semantic answer types and serializer

**User story:** As the grader interface, I want typed semantic values converted into exact strings without generative rewriting.

**Learn:** Discriminated unions, `Decimal`, formatting invariants, and exact-match risk.

**Build:** Serializers for every contract type, including list policies, entity components, currency, and notation.

**Acceptance:** Golden tests cover punctuation, symbols, accents, ordering, precision, separators, and absence of explanation.

**Dependencies:** A03, A11.

### [x] GAIA-A14 — Implement final answer validators

**User story:** As the TaskGraph, I want invalid serialized answers rejected before checkpoint acceptance.

**Learn:** Structural validation and negative testing.

**Build:** Checks for type, precision, sorting, duplicates, Markdown/prose, forbidden prefixes, and contract-specific shape.

**Acceptance:** Poisoned strings are rejected with actionable reason codes; valid edge cases remain unchanged.

**Dependencies:** A13.

## Epic A5 — LangGraph control plane and persistence

### [x] GAIA-A15 — Build a minimal typed TaskGraph

**User story:** As a learner, I want the smallest complete LangGraph task flow before specialist complexity is introduced.

**Learn:** State, nodes, direct edges, conditional edges, reducers, compilation, and invocation.

**Build:** Prepare → analyze → route placeholder → solve placeholder → verify placeholder → serialize → end.

**Acceptance:** The graph runs on synthetic tasks, renders as Mermaid, and state transitions pass tests.

**Dependencies:** A03, A11, A13, A14.

### [x] GAIA-A16 — Add SQLite checkpointing and resume

**User story:** As the operator, I want a failed process to resume from its last successful graph step.

**Learn:** LangGraph threads, checkpoints, replay, idempotency, and SQLite WAL.

**Build:** Async SQLite checkpointer, task thread IDs, checkpoint inspection, and resume commands.

**Acceptance:** An intentionally failing node resumes without repeating completed side effects; state history is inspectable.

**Dependencies:** A15.

### [x] GAIA-A17 — Implement deterministic solver routing

**User story:** As the TaskGraph, I want hard signals and known profiles to choose specialist solvers predictably.

**Learn:** Registry pattern, capability scoring, and conditional graph edges.

**Build:** Solver interface, registry, route policy, and deep-research fallback.

**Acceptance:** Attachment extensions, YouTube wording, historical constraints, table shapes, and known profiles route correctly.

**Dependencies:** A06, A12, A15.

### [x] GAIA-A18 — Build the RunGraph and task dispatch

**User story:** As the operator, I want one run to coordinate all tasks while preserving per-task recovery.

**Learn:** Graph composition, subgraphs, parallel branches, reducers, and concurrency limits.

**Build:** Snapshot guard, task selection, TaskGraph dispatch, result collection, and resume selection.

**Acceptance:** Synthetic 20-task runs complete with bounded concurrency; one failed task does not erase accepted results.

**Dependencies:** A05, A16, A17.

### [x] GAIA-A19 — Implement dry-run reporting and submission preflight skeleton

**User story:** As the operator, I want to inspect completeness and risk without ever submitting accidentally.

**Learn:** Safety gates and manifest-based workflows.

**Build:** `--dry-run`, `--no-submit`, task summary, preflight interfaces, and frozen manifest model.

**Acceptance:** No code path can call `/submit`; incomplete, duplicate, unknown, or empty synthetic results fail preflight.

**Dependencies:** A14, A18.

## Milestone A gate

```text
[x] Current questions can be fetched and hashed.
[x] Current attachments can resolve through official primary/fallback paths.
[x] Task profiles and contracts are typed and answer-free.
[x] TaskGraph and RunGraph checkpoint and resume.
[x] Serialization fixtures are exact.
[x] Dry mode cannot submit.
```

# Milestone B — Deterministic solvers

### [x] GAIA-B01 — Implement transformed-text solver

**Learn:** Pure functions and invariants. **Acceptance:** Synthetic transforms round-trip and preserve exact characters.

### [x] GAIA-B02 — Implement operation-table parser and solver

**Learn:** Complete parsing and exhaustive algorithms. **Acceptance:** Every ordered pair is checked; malformed/incomplete tables fail.

### [x] GAIA-B03 — Implement structured sports/table reduction

**Learn:** Numeric coercion, argmax/min, row integrity, and tie rules. **Acceptance:** Synthetic full tables reproduce max/min and requested same-row fields.

### [x] GAIA-B04 — Implement constrained Python inspection and execution

**Learn:** ASTs, subprocess isolation, resource limits, and deterministic replay. **Acceptance:** Safe fixtures run; network/subprocess/timeout fixtures are blocked; simple output is independently confirmed.

### [x] GAIA-B05 — Implement workbook inspection

**Learn:** XLSX internals, sheets, formulas, hidden rows, formats, and cached values. **Acceptance:** A workbook audit report identifies all relevant structural features.

### [x] GAIA-B06 — Implement independent spreadsheet calculations

**Learn:** pandas versus `openpyxl`, `Decimal`, classification audits, and parity. **Acceptance:** Both paths agree exactly on synthetic food/drink workbooks and detect double-counted totals.

### [x] GAIA-B07 — Integrate deterministic verification policies

**Learn:** Policy-based verification. **Acceptance:** Each B solver can be accepted without an LLM only when its deterministic authority passes.

### [x] GAIA-B08 — Add property and mutation tests

**Learn:** Hypothesis and poisoned implementations. **Acceptance:** Tests catch off-by-one, tie, ordering, floating-point, and incomplete-table mutations.

## Milestone B gate

All deterministic synthetic fixtures pass, spreadsheet paths agree, and no deterministic result depends on model voting.

**Gate status:** Passed. B01–B08 have deterministic unit, replay-verification,
property, and poisoned-implementation coverage using synthetic answer-free inputs.

# Milestone C — Web, historical, scholarly, and cross-lingual research

### [x] GAIA-C01 — Define search and page-retrieval provider interfaces

**Learn:** Ports/adapters and provider independence. **Acceptance:** Two mock providers are interchangeable and results carry provenance.

### [x] GAIA-C02 — Implement primary and secondary search paths

**Learn:** Query formulation, quoted/site/native searches, caching, and rate limits. **Acceptance:** Provider failure routes to a genuinely separate backend.

### [x] GAIA-C03 — Implement the page retrieval ladder

**Learn:** HTTP, structured extraction, readers, browser fallback, and archives. **Acceptance:** Static, JS-dependent, blocked, and archived fixtures follow the correct path.

### [x] GAIA-C04 — Add prompt-injection detection and source boundaries

**Learn:** Untrusted content handling. **Acceptance:** Injected pages cannot alter tool policy, reveal secrets, or change the objective.

### [x] GAIA-C05 — Implement MediaWiki historical revision client

**Learn:** MediaWiki API, revision timestamps, old content, and links. **Acceptance:** Mock histories select the latest revision at or before an inclusive cutoff.

### [x] GAIA-C06 — Implement historical Wikipedia extraction solver

**Learn:** Wikitext parsing and deterministic dated extraction. **Acceptance:** Synthetic discography and nomination fixtures use the requested revision, not current content.

### [x] GAIA-C07 — Implement archive and temporal validity checks

**Learn:** Content-state time versus retrieval time. **Acceptance:** Current undated evidence is rejected for historical-state claims; valid retrospective evidence can pass.

### [x] GAIA-C08 — Implement PDF extraction ladder

**Learn:** PyMuPDF, pypdf, page rendering, OCR, and exact context windows. **Acceptance:** Text, layout-sensitive, and scanned fixtures yield page-addressable evidence.

### [x] GAIA-C09 — Add LlamaIndex supporting document retrieval

**Learn:** Loading, chunking, metadata, index, QueryEngineTool, and evaluation. **Acceptance:** It finds relevant synthetic passages, while exact extraction remains required before approval.

### [x] GAIA-C10 — Implement scholarly evidence solver

**Learn:** DOI resolution, acknowledgments, specimen/depository language, and source hierarchy. **Acceptance:** Synthetic papers distinguish funding relationships and deposition from author affiliation.

### [x] GAIA-C11 — Implement cross-lingual entity research utilities

**Learn:** Native queries, stable identifiers, namesakes, and source-backed Romanization. **Acceptance:** Ambiguous synthetic entities require disambiguating fields before selection.

### [x] GAIA-C12 — Implement structured web table solver

**Learn:** Extract-compute-verify instead of snippet trust. **Acceptance:** Historical roster/recipient/table fixtures reproduce uniqueness and ties.

### [x] GAIA-C13 — Store claim-level evidence and provenance

**Learn:** Evidence graphs and auditability. **Acceptance:** Every nondeterministic candidate references evidence that directly supports its claim.

## Milestone C gate

Historical fixtures use date-valid sources, scholarly fixtures use primary passages, and search snippets cannot independently approve an answer.

**Gate status:** Passed. C01-C13 cover historical, scholarly, cross-lingual,
structured-table, claim-evidence, and guarded long-document retrieval paths. C09 uses
real BM25 plus optional injected dense embeddings and preserves page/artifact provenance;
exact source evidence, not retrieval score, remains authoritative.

# Milestone D — Audio and video

### [x] GAIA-D01 — Build media probing and normalization

**Learn:** `ffprobe`, codecs, sample rates, duration, and immutable derived artifacts. **Acceptance:** Audio/video fixtures normalize reproducibly with timestamps preserved.

### [x] GAIA-D02 — Implement timestamped primary ASR

**Learn:** Speech decoding and segment confidence. **Acceptance:** Synthetic speech returns timestamped segments and normalized text without losing raw transcript.

### [x] GAIA-D03 — Add second-decode and disputed-segment workflow

**Learn:** Independent extraction and phonetic ambiguity. **Acceptance:** Disputed numeric/exact-word segments trigger a focused second pass.

### [x] GAIA-D04 — Implement ingredient extraction mode

**Learn:** Section boundaries, entity extraction, exclusions, and conservative normalization. **Acceptance:** Synthetic recipes exclude crust/measurements and never invent conventional ingredients.

### [x] GAIA-D05 — Implement numeric-page extraction mode

**Learn:** Number-word parsing, teen/tens ambiguity, deduplication, and ordering. **Acceptance:** Synthetic audio page lists serialize numerically and flag ambiguous spans.

### [x] GAIA-D06 — Implement YouTube acquisition and captions

**Learn:** Video IDs, captions, media download, caching, and fallback. **Acceptance:** Mock/direct caption and audio fallback paths are independently tested.

### [x] GAIA-D07 — Implement exact YouTube speech solver

**Learn:** Anchor matching, speaker/utterance boundaries, and verbatim adjudication. **Acceptance:** Synthetic clips return exact replies only when captions and audio evidence are reconciled.

### [x] GAIA-D08 — Implement frame sampling and contact sheets

**Learn:** FPS, scene changes, timestamps, image grids, and artifact references. **Acceptance:** Synthetic videos produce complete coarse coverage and dense requested windows.

### [x] GAIA-D09 — Define structured VLM frame analysis

**Learn:** VLMs as sensors, structured species sets, bounding regions, and uncertainty. **Acceptance:** VLM output validates against schema and cannot return only an unsupported number.

### [x] GAIA-D10 — Implement simultaneous-species video solver

**Learn:** Temporal overlap, set counting, dense resampling, and avoiding duplicate species. **Acceptance:** Synthetic videos with brief overlap and repeated individuals yield the true maximum.

### [x] GAIA-D11 — Add independent vision pass and disagreement handling

**Learn:** Model/source diversity and targeted adjudication. **Acceptance:** Top candidate windows are inspected independently; disagreement increases resolution/sampling rather than averaging.

## Milestone D gate

Exact speech has timestamp evidence, numeric audio survives ambiguity fixtures, and visual questions cannot be answered from transcripts alone.

**Gate status:** Passed. Real FFmpeg fixtures prove reproducible mono/16 kHz PCM
normalization and timestamped frame/contact-sheet extraction. Timestamped ASR,
caption reconciliation, conservative audio extraction, structured species sets,
and independent visual disagreement policies pass answer-free fixtures.

# Milestone E — Chess

### [x] GAIA-E01 — Build chess image preprocessing and orientation checks

**Learn:** Board coordinates and image normalization. **Acceptance:** Synthetic boards preserve square mapping under expected orientations.

### [x] GAIA-E02 — Implement two structured board-transcription paths

**Learn:** FEN, piece-square schemas, and independent sensing. **Acceptance:** Each path outputs complete occupancy and disagreement locations.

### [x] GAIA-E03 — Reconcile and validate candidate FENs

**Learn:** `python-chess`, kings, side to move, legal positions, and reconciliation. **Acceptance:** Illegal or visually inconsistent states cannot reach the engine.

### [x] GAIA-E04 — Integrate Stockfish and optional tablebases

**Learn:** Engine analysis, depth/time, evaluation, and alternative comparison. **Acceptance:** Known tactical fixtures produce verified best moves with evidence.

### [x] GAIA-E05 — Implement deterministic SAN serialization

**Learn:** UCI versus SAN and board-dependent notation. **Acceptance:** SAN is generated by `python-chess` and parses as legal on the validated position.

## Milestone E gate

Known diagrams pass, VLM-only moves are impossible, and every accepted chess answer has a validated board plus engine result.

**Gate status:** Passed. Synthetic orientation/transcription fixtures and a real
local Stockfish 18 UCI run prove legal winning-move selection and SAN round-trip.

# Milestone F — Research agents, verification, and observability

### [x] GAIA-F01 — Implement the smolagents web researcher

**Learn:** Tools, ToolCallingAgent, ReAct traces, limits, and planning. **Acceptance:** The worker uses narrow tools and returns structured research packets.

### [x] GAIA-F02 — Implement deterministic research supervision and replanning

**Learn:** Planning intervals, unresolved claims, and bounded autonomy. **Acceptance:** Weak first searches produce changed targeted objectives within step budgets. A managed-agent hierarchy remains deliberately postponed until evals justify its extra failure surface.

### [ ] GAIA-F03 — Add safe optional CodeAgent backend

**Learn:** Code actions and executor safety. **Acceptance:** CodeAgent is enabled only with an approved isolated executor; default mode remains safe.

### [x] GAIA-F04 — Build structural and deterministic verifier nodes

**Learn:** Verification ladders and conditional routing. **Acceptance:** Failed hard checks route to repair/block without model override.

### [x] GAIA-F05 — Build source and temporal verifier nodes

**Learn:** Claim support, entity identity, and historical validity. **Acceptance:** Poisoned evidence packets are rejected with precise reason codes.

### [x] GAIA-F06 — Build independent retrieval verifier

**Learn:** Independence beyond repeated sampling. **Acceptance:** Verification uses a different source/query/provider/extraction method where required.

### [x] GAIA-F07 — Build independent model critic

**Learn:** Structured criticism and model diversity. **Acceptance:** The critic returns approve/reject/uncertain with missing evidence, never an ungrounded vote.

### [x] GAIA-F08 — Build evidence-first adjudicator

**Learn:** Conflict resolution. **Acceptance:** Conflicts prefer deterministic and primary/date-valid evidence or request targeted research; no majority vote.

### [x] GAIA-F09 — Implement evidence-derived risk scoring

**Learn:** Feature-based confidence and adaptive compute. **Acceptance:** Hard failures remain blocking; high-risk features trigger additional verification.

### [x] GAIA-F10 — Add OpenTelemetry and Langfuse instrumentation

**Learn:** Traces, spans, latency, tokens, costs, and redaction. **Acceptance:** A synthetic run shows graph/model/tool spans with no secrets or gated answer fields.

### [ ] GAIA-F11 — Build the pre-submission risk dashboard

**Learn:** Operational decision support. **Acceptance:** Every task shows route, candidate status, evidence class, deterministic checks, and residual risk.

### [ ] GAIA-F12 — Add retry, chaos, and recovery tests

**Learn:** Exponential backoff, idempotency, fault injection, and resume. **Acceptance:** Provider/API/process failures recover without redoing accepted work or duplicating side effects.

## Milestone F gate

Poisoned candidates are rejected, high-risk tasks receive independent paths, traces are safe, and restarts preserve progress.

**Hybrid gate status:** F01-F02, F04-F10 pass unit, adversarial, and end-to-end
integration tests. LangGraph remains the global control plane; smolagents is isolated to
fallback research; LlamaIndex passages must resolve through the evidence catalog; and
traces reject prompt, answer, token, and credential fields. F03 and F11-F12 remain
deliberately postponed breadth.

# Milestone G — Clean current evaluation dry run

### [x] GAIA-G01 — Run all current public questions without gold answers

**Acceptance:** Every current task produces READY or a structured BLOCKED result; no official submission occurs.

### [x] GAIA-G02 — Resolve all acquisition and execution failures

**Acceptance:** No attachment, provider, parser, ASR, video, engine, or checkpoint infrastructure failure remains.

### [x] GAIA-G03 — Review all temporal and source risks

**Acceptance:** No historical task is supported only by a current undated page; scholarly tasks use primary evidence where obtainable.

### [x] GAIA-G04 — Review all modality risks

**Acceptance:** Speech has timestamps, video has frame evidence, chess has engine evidence, and XLSX has parity.

### [x] GAIA-G05 — Review all output contracts

**Acceptance:** All serialized answers contain answer-only strings and pass question-specific contracts.

### [x] GAIA-G06 — Freeze the candidate run and manifest

**Acceptance:** Exactly 20 unique expected IDs are represented; the run, question snapshot, software, profiles, and payload are hashed.

## Milestone G gate

The risk dashboard has no unresolved engineering failure and the frozen payload passes the complete preflight without using target gold answers.

**Current gate status:** Passed. All 20 tasks are READY with valid serialization and
reviewed evidence. The official gated fallback resolved the five missing course-endpoint
attachments without accessing target answers. Preflight froze exactly 20 unique IDs
against snapshot `e03d1d62…ca148`; submission remains disabled and was not attempted.

# Milestone H — Deployment, certificate, and deliberate submission

### [x] GAIA-H01 — Create the Docker Hugging Face Space

**Learn:** Docker Spaces, system dependencies, secrets, and Gradio. **Acceptance:** Public code builds cleanly and private runtime data is not exposed.

**Local status:** Dockerfile, Gradio 6 UI, Space metadata, dependency extras, non-root
runtime, and a private-data-denying Docker context are implemented. A real Docker image
build completed, and the resulting container served the Gradio UI with HTTP 200 on
localhost. Remote Space creation remains a separate external publishing action.

### [ ] GAIA-H02 — Build authenticated operator UI

**Learn:** Gradio events and HF login. **Acceptance:** Only the operator can start expensive runs or submission; public users can inspect safe project information.

### [x] GAIA-H03 — Add explicit submission client and LangGraph interrupt

**Learn:** Human-in-the-loop and frozen side effects. **Acceptance:** Submission requires config gate, preflight, owner approval, and resumes with the frozen payload.

**Status:** The isolated no-retry client, configuration gate, exact-count preflight,
hash-bound owner approval, and checkpointed LangGraph interrupt/resume are implemented
and tested. No submission command is exposed.

### [ ] GAIA-H04 — Perform a clean remote dry run

**Acceptance:** The deployed environment resolves all dependencies, providers, attachments, and tasks with checkpoints and traces.

### [x] GAIA-H05 — Publish final inspectable documentation

**Acceptance:** README explains architecture, legitimate methodology, setup, dry run, testing, and public code without exposing answers or gated data.

### [x] GAIA-H06 — Deliberately submit and store the receipt

**Acceptance:** The approved manifest is submitted once; aggregate receipt and hashes are stored privately; no task answers are published through logs.

### [ ] GAIA-H07 — Complete certificate steps

**Acceptance:** The relevant Unit 1/use-case requirements are complete, the final score is recorded, and the official certificate page recognizes the account.

## Milestone H gate

The public code is inspectable, the official score is recorded, and the course certificate requirements have been completed.

# Cross-cutting definition of done

Every story must meet the collaboration checklist in [03_LEARNING_WORKFLOW.md](03_LEARNING_WORKFLOW.md). In particular:

- Tests accompany implementation.
- Changed code is explained at beginner level.
- No gold answers or answer leaks enter code, fixtures, traces, or documentation.
- Provider output is validated before entering graph state.
- Failure behavior is explicit.
- Documentation and backlog status remain synchronized.
