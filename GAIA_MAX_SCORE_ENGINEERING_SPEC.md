# GAIA Maximum-Score Engineering Specification
## Hugging Face Agents Course — Unit 4 Final Assignment
**Research snapshot:** 2026-08-26
**Objective:** Design an agent system for the highest realistically achievable score on the *current* Hugging Face Agents Course GAIA evaluation, aiming for 20/20 rather than merely the 30% certificate threshold.

> This is an engineering requirements document, not a benchmark-answer sheet. It deliberately contains no GAIA gold answers. It uses the public evaluation prompts to determine required capabilities and failure modes.

---

# 0. Codex directive

Treat this document as the source of truth for implementation.

The target is not “make a nice general-purpose chatbot.” The target is:

> Build a highly reliable, reproducible, evidence-grounded solver for the current 20-question Hugging Face Agents Course evaluation, with deterministic specialist solvers wherever possible, independent verification for fragile tasks, historical-source awareness, robust attachment retrieval, and exact-match-safe output serialization.

Optimize in this order:

1. **Correctness**
2. **Evidence quality / verification**
3. **Exact output compliance**
4. **Reliability under network/API failures**
5. **Reproducibility and observability**
6. Cost
7. Latency

Do **not** simplify architecture merely to resemble an introductory agent tutorial. Complexity is acceptable when it measurably reduces failure probability.

Do **not** hard-code benchmark answers, scrape answer leaks, or read the target questions' `Final answer` field during development. This should remain a real agentic solution that can be publicly inspected.

---

# 1. Executive conclusion

A maximum-score solution should **not** be a single unconstrained ReAct agent with a bag of tools.

The current evaluation is sufficiently small and structurally stable that the best engineering strategy is a **deterministic outer orchestration graph with specialized modality/domain solvers**, plus a strong open-ended research agent as a fallback.

Recommended architecture:

```text
Official question API
        |
        v
Question Snapshot + Integrity Check
        |
        v
Attachment Resolver
(API -> official gated GAIA dataset fallback -> cache)
        |
        v
Task Analyzer / Output-Contract Parser
        |
        v
Deterministic Router
        |
        +-----------------------+
        |                       |
        v                       v
Specialized solver         Deep-research solver
(text/code/xlsx/audio/     (historical web,
video/chess/table/etc.)     scholarly, cross-lingual)
        |                       |
        +-----------+-----------+
                    |
                    v
           Candidate Answer(s)
                    |
                    v
           Verification Ladder
     deterministic -> source/date ->
     independent retrieval -> model critic
                    |
                    v
         Structured Answer Object
                    |
                    v
      Question-Aware Serializer
                    |
                    v
        Final Contract Validator
                    |
                    v
       Per-Task Checkpoint/Trace
                    |
                    v
       20-answer Submission Preflight
                    |
                    v
          Official /submit
```

The most important design principle is:

> **LLMs should decide/search/reason where necessary; deterministic software should parse, calculate, execute, sort, validate, and format whenever possible.**

A 20/20 attempt requires reducing *correlated* failure. A second LLM repeating the same flawed research path is not sufficient verification. Prefer an independent source, independent extraction method, or domain-native deterministic checker.

---

# 2. Official evaluation contract — hard requirements

## 2.1 Current task selection

The current scoring backend:

- loads `gaia-benchmark/GAIA`
- configuration: `2023_level1`
- split: `validation`
- keeps tasks whose annotated:
  - **Number of tools < 3**
  - **Number of steps < 6**

The current public `/questions` endpoint exposes **20 tasks**.

This is more specific than “Level 1 GAIA.” The target is a selected low-tool/low-step subset.

### Engineering implication

The system should fetch `/questions` at runtime and create a snapshot:

```json
{
  "retrieved_at": "...",
  "count": 20,
  "task_ids": ["..."],
  "questions_sha256": "...",
  "task_profiles_version": "..."
}
```

If the question count or hash changes from the development snapshot, do not silently submit. Re-run task profiling first.

---

## 2.2 Official routes

The course exposes:

```text
GET  /questions
GET  /random-question
GET  /files/{task_id}
POST /submit
```

Expected submission shape:

```json
{
  "username": "HF_USERNAME",
  "agent_code": "https://huggingface.co/spaces/<user>/<space>/tree/main",
  "answers": [
    {
      "task_id": "...",
      "submitted_answer": "..."
    }
  ]
}
```

The Space/code should be public for inspection.

---

## 2.3 Exact-match behavior

The live scoring backend currently compares:

```python
submitted.strip().lower() == ground_truth.strip().lower()
```

Therefore:

- leading/trailing whitespace is ignored
- capitalization is ignored
- punctuation is **not** ignored
- thousands separators are **not** ignored
- `$`, `%`, units, articles, extra explanation, prefixes, list ordering, extra decimals, and alternate spellings can all turn a semantically correct answer into zero points

The score denominator is the full filtered set, not merely successful attempts:

```text
score = correct_count / number_of_ground_truth_tasks
```

So crashes/skips effectively cost points.

### Hard output rule

Never submit:

```text
FINAL ANSWER: ...
Answer: ...
I found that ...
```

Submit only the requested answer string.

---

## 2.4 Duplicate and invalid IDs

The scorer skips duplicate task IDs and unknown IDs. The submission preflight should therefore require:

- exactly one answer for each currently fetched task ID
- no duplicate IDs
- no unknown IDs
- no empty answers
- all submitted answers converted to strings
- exact count equals current `/questions` count

---

## 2.5 Best-score behavior

The current scorer keeps a user's improved/highest score rather than intentionally replacing it with a lower later score. Nevertheless, official submissions should not be used as the main debugging loop because the API returns aggregate correctness, not task-by-task correctness.

---

# 3. Critical operational issue: attachments

## 3.1 Current breakage

As of the 2026 research snapshot, Hugging Face community reports confirm that the official:

```text
/files/{task_id}
```

endpoint can return 404 / “No file path associated” for attachment questions after a GAIA dataset layout change.

A July 2026 report reproduced this against the exact Excel task currently in the course evaluation.

Five of the current 20 tasks use attachments. Losing attachment access can therefore make 20/20 impossible and can immediately sacrifice about 25 percentage points.

## 3.2 Required attachment resolver

Implement:

```text
resolve_attachment(task):
    if task.file_name is empty:
        return None

    1. Try official /files/{task_id}
       - timeout
       - retries
       - validate non-error MIME and nonzero bytes

    2. If unavailable:
       use authenticated Hugging Face access to
       gaia-benchmark/GAIA
       and resolve the file from the official gated dataset

    3. Cache bytes by:
       task_id + filename + SHA256

    4. Validate:
       expected extension
       MIME/magic bytes
       SHA256
       safe filename
       file size bounds

    5. Return immutable local path + metadata
```

### Required environment

```text
HF_TOKEN=<token with accepted GAIA dataset access>
```

### Do not use as the preferred source

There are community mirrors of GAIA attachments. They may be useful diagnostically, but the production solution should prefer the **official gated dataset** for benchmark hygiene and provenance.

---

# 4. Current 20-task capability matrix

This matrix is based on the public live `/questions` endpoint on 2026-08-26. It intentionally does not contain answers.

| # | Public task ID | Task shape | Primary solver | Independent verification | Output contract / main risk |
|---|---|---|---|---|---|
| 1 | `8e867cd7-cff9-4e6c-867a-ff5ddc2550be` | Historical Wikipedia discography count, specifically latest 2022 English Wikipedia version | MediaWiki revision-aware extractor + deterministic date/count logic | Archived/revision page cross-check | Integer only; current page may differ from 2022 |
| 2 | `a1e91b78-d3d8-4675-bb8d-62741b4b68a6` | YouTube visual question: maximum simultaneous bird species on camera | Video download/frames + temporal visual analysis | Dense resampling around candidate maxima + second VLM/manualizable contact sheet | Integer; transcript alone cannot solve it |
| 3 | `2d83110e-a098-4ebb-9987-066c06fa42d0` | Reversed/simple transformed text | Pure deterministic string transform | Round-trip transform invariant | Exact string |
| 4 | `cca530fc-4052-43b2-b130-b30968d8aa44` | Chessboard PNG, black to move, winning move | Board-to-FEN transcription + `python-chess` + Stockfish | Multiple board transcriptions + legal-state validation + engine confirmation | SAN/algebraic notation; VLM direct answer unsafe |
| 5 | `4fc2f1ae-8625-45b5-ab34-ad4433bc21f8` | Identify nominator(s) of only dinosaur Featured Article promoted in Nov 2016 | Wikipedia FA log/history + FAC nomination page | Revision/history and nomination page consistency | Person name(s), exact list formatting |
| 6 | `6f37996b-2ac7-44b0-8e68-6d28256631b4` | Binary-operation table, identify elements participating in noncommutativity counterexamples | Deterministic table parser/Python | Exhaustive pair check | Alphabetical comma-separated list |
| 7 | `9d191bce-651d-4746-be2d-7ef8ecadb9c2` | YouTube exact spoken reply after anchor phrase | Timestamped captions/transcript + ASR around anchor | Independent audio transcription ± nearby seconds | Exact quotation content, no paraphrase |
| 8 | `cabe07ed-9eca-40ea-8ead-410ef5e83f91` | Historical LibreTexts textbook detail, compile date specified | Site-targeted search + archived/version-aware page extraction | Search exact section/title + archive/source metadata | Surname only |
| 9 | `3cef3a44-215e-4aed-8e3b-b1e3f08063b7` | Botanical classification of listed foods, exclude botanical fruits | Item-by-item ontology/classification + deterministic filtering | Trusted botanical references for ambiguous items | Alphabetical comma list |
|10 | `99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3` | MP3 recipe; filling ingredients only, measurements removed | High-quality ASR + ingredient extraction | Second ASR/segment review + recipe-structure check | Alphabetize; exclude crust; no measurements |
|11 | `305ac316-eef6-4446-960a-92d80d542f82` | Cross-lingual TV actor/entity chain (Polish adaptation → another show role) | Bilingual entity research | Second independent source / entity-ID disambiguation | First name only |
|12 | `f918266a-b3e0-4914-865d-4faa564f1aef` | Attached Python program; determine final numeric output | Sandboxed execution | Static/source inspection or independent calculation | Exact numeric output |
|13 | `3f57289b-8c60-48be-bd80-01f8099ca449` | 1977 Yankees batting table; max walks then same-row at-bats | Structured stats extraction + Python argmax | Second stats source / tie assertion | Integer |
|14 | `1f975693-876d-457b-a649-393859e79bf3` | MP3 homework instruction; extract page numbers | Timestamped ASR + number parser | Second transcription / segment replay | Numeric ascending comma list |
|15 | `840bfca7-4f7b-481a-8794-c560c340185d` | Date-specific Universe Today article → linked paper → NASA award supporting named author | Article link traversal + primary paper acknowledgments | Paper PDF/text + metadata cross-check | Award identifier exact spelling |
|16 | `bda648d7-d618-4883-88f4-3466eabd860e` | Scholarly paper about Vietnamese specimens; eventual deposition location | PDF/scholarly search + exact specimen/depository extraction | Institution lookup + paper passage | City only |
|17 | `cf106601-ab4f-4af9-b045-5295fe67b37d` | 1928 Olympics least-athlete country; alphabetical tie-break; IOC code | Structured table + deterministic min/tie sort + IOC mapping | Secondary historical sports source | IOC code only |
|18 | `a0c07678-e491-4bbc-8f0b-07405144218f` | July-2023 Japanese baseball jersey-number neighbors of a named pitcher | Date-aware Japanese/native roster research | Archived/native source + secondary roster; verify position and transliteration | Roman-character surnames |
|19 | `7bd855d8-463d-4ed5-93ca-5fe35145f733` | XLSX sales: total food sales excluding drinks | `openpyxl` + pandas + Decimal | Independent parser/recomputation and row classification audit | USD with exactly two decimals as requested |
|20 | `5a0c1adf-205e-4841-a666-7c3ef95def9d` | Malko Competition recipients after 1977; nationality corresponds to country no longer existing | Structured recipient list + historical-country reasoning | Independent nationality/source validation + uniqueness assertion | First name only |

## 4.1 Distribution

Current set includes at least:

- **5 attachment tasks**
  - 1 image/chess
  - 2 audio
  - 1 Python source
  - 1 Excel workbook
- **2 YouTube tasks**
  - one genuinely visual
  - one exact-dialogue/audio task
- multiple date/version-sensitive web tasks
- multiple structured-table/deterministic tasks
- multiple strict list/name/number formatting tasks

This distribution strongly favors specialization.

---

# 5. Why a generic ReAct agent is not enough

A generic tool-calling loop can score reasonably, but full-grade attempts are vulnerable to several failure classes.

## 5.1 “Correct reasoning, wrong serialization”

Observed in public implementations.

Example failure class:

```text
computed semantic value correctly
        |
        v
LLM adds currency symbol / comma / explanation
        |
        v
exact-match failure
```

Therefore formatting cannot be an informal final prompt. It must be an explicit output contract and deterministic serializer.

---

## 5.2 “Right source, wrong temporal version”

Several current tasks explicitly say things like:

- latest version in 2022
- as of July 2023
- article on a specific date
- textbook compiled on a specified date
- event in a particular historical month/year

A current webpage can be factually true today but still be the wrong evidence.

Temporal constraints must be parsed as first-class task metadata.

---

## 5.3 “Vision confidence without verification”

Princeton's HAL GAIA analysis shows an instructive failure pattern on the current chess-image task: a strong VLM can confidently produce a chess move after visually inspecting the board while never constructing/verifying a legal FEN or consulting a chess engine.

For chess:

```text
VLM = transcription sensor
Stockfish/python-chess = verifier/decision authority
```

not:

```text
VLM = chess engine
```

The same principle generalizes:

- spreadsheet → parser/calculator
- code → interpreter
- operation table → exhaustive Python
- sports min/max → deterministic table reduction
- video count → frames and temporal verification
- exact speech → audio/transcript segment, not semantic paraphrase

---

## 5.4 “Clean tool path but false premise”

Research agents can find a source that appears relevant, infer a plausible bridge, and then become highly confident. Confidence is not proof.

A verifier must ask:

1. Is the source actually about the same entity?
2. Is the source date/version correct?
3. Does the cited passage directly establish the claim?
4. Did we answer the requested field, not a nearby field?
5. Is there a tie/uniqueness condition?
6. Can a second, independent route falsify this?

---

## 5.5 “Unavailable source → invented fallback”

When web pages, captions, or papers are inaccessible, frontier agents can still produce plausible-looking answers.

The architecture must distinguish:

```text
NOT FOUND
FOUND INDIRECTLY
FOUND IN PRIMARY SOURCE
VERIFIED BY SECOND SOURCE
```

A model should never silently convert `NOT FOUND` into high-confidence factual output.

---

# 6. Recommended architecture

## 6.1 Outer control plane

Use either:

- **LangGraph** for explicit states, retries, branches, checkpoints; or
- a custom typed Python state machine/DAG if you want maximum transparency.

For this benchmark I prefer:

> **LangGraph/custom deterministic orchestration outside + specialized Python tools + a smolagents/Open-Deep-Research-inspired sub-agent for difficult web research.**

The outer graph should *not* let the LLM decide every transition.

### Suggested nodes

```text
START
  |
  v
fetch_question_snapshot
  |
  v
resolve_attachment
  |
  v
analyze_task
  |
  v
route_task
  |
  +--> deterministic_text_solver
  +--> code_solver
  +--> spreadsheet_solver
  +--> audio_solver
  +--> youtube_speech_solver
  +--> youtube_visual_solver
  +--> chess_solver
  +--> wikipedia_history_solver
  +--> scholarly_solver
  +--> structured_web_table_solver
  +--> crosslingual_research_solver
  +--> deep_research_fallback
  |
  v
verify_candidate
  |
  +-- reject --> targeted_retry / alternate solver
  |
  v
serialize_answer
  |
  v
validate_output_contract
  |
  v
checkpoint
  |
  v
END
```

---

## 6.2 Why combine deterministic routing and agentic research?

Public high-scoring student implementations and Hugging Face's own Open Deep Research suggest two complementary strengths:

### Agentic planning/research is excellent for

- unknown web paths
- entity disambiguation
- multi-hop paper/article research
- query reformulation
- deciding which primary source to inspect
- bilingual search

### Deterministic routing/solvers are better for

- Python
- Excel
- algebraic operation tables
- sorting
- exact list manipulation
- date comparison
- chess legality/engine verification
- output formatting
- submission integrity

Maximum score comes from using both.

---

# 7. Core state/data models

Use typed objects. Do not pass giant unstructured chat histories between all nodes.

## 7.1 TaskProfile

```python
@dataclass
class TaskProfile:
    task_id: str
    question: str
    file_name: str | None
    modality: Literal[
        "text", "image", "audio", "python", "xlsx",
        "youtube_audio", "youtube_visual", "web"
    ]
    task_class: str
    temporal_constraint: str | None
    required_sources: list[str]
    output_contract: "OutputContract"
    risk_flags: list[str]
```

## 7.2 OutputContract

```python
@dataclass
class OutputContract:
    answer_type: Literal[
        "integer", "decimal", "string", "name",
        "first_name", "surname", "city", "ioc_code",
        "list", "currency", "chess_san"
    ]

    sort: Literal["none", "alphabetical", "numeric_ascending"] = "none"
    delimiter: str = ", "
    decimal_places: int | None = None

    units_policy: Literal[
        "omit", "literal_required", "question_specific"
    ] = "omit"

    include_currency_symbol: bool | None = None
    thousands_separator: bool = False

    name_scope: Literal[
        "full", "first", "surname", "unchanged"
    ] = "unchanged"

    notation: str | None = None
    trailing_punctuation: bool = False
    expected_item_count: int | None = None
```

## 7.3 Evidence

```python
@dataclass
class Evidence:
    evidence_id: str
    claim: str
    source_url: str | None
    source_type: str
    retrieved_at: datetime
    source_date: str | None
    snapshot_date: str | None
    is_primary_source: bool
    excerpt: str | None
    extraction_method: str
    confidence: float
```

## 7.4 CandidateAnswer

```python
@dataclass
class CandidateAnswer:
    semantic_value: object
    solver: str
    evidence_ids: list[str]
    deterministic_checks: list[str]
    unresolved_risks: list[str]
    confidence: float
```

## 7.5 VerificationResult

```python
@dataclass
class VerificationResult:
    approved: bool
    passed_checks: list[str]
    reason_codes: list[str]
    critique: str
    requested_retry: str | None
    evidence_conflicts: list[str]
```

Store concise rationales/evidence, not private model chain-of-thought.

---

# 8. Task analyzer

The analyzer should extract the *contract* before trying to answer.

Example conceptual output:

```json
{
  "task_class": "historical_wikipedia",
  "temporal_constraint": {
    "kind": "revision_cutoff",
    "end": "2022-12-31"
  },
  "requested_operation": "count",
  "filters": {
    "year_start": 2000,
    "year_end": 2009,
    "inclusive": true,
    "category": "studio_album"
  },
  "output_contract": {
    "answer_type": "integer"
  }
}
```

### Required analyzer flags

- historical snapshot requested?
- exact quote requested?
- “as of” date?
- sorting requested?
- tie-breaking rule?
- include/exclude conditions?
- answer scope: first name / surname / city / code?
- specified notation?
- units requested or prohibited?
- attachment present?
- visual evidence actually necessary?
- calculation possible deterministically?
- uniqueness claim (“only”, “least”, “highest”) that requires exhaustive checking?

---

# 9. Router

Use deterministic rules first.

Pseudo-policy:

```python
if attachment_ext == ".py":
    route = "code_solver"

elif attachment_ext in {".xlsx", ".xls", ".csv"}:
    route = "spreadsheet_solver"

elif attachment_ext in {".mp3", ".wav", ".m4a"}:
    route = "audio_solver"

elif attachment_ext in {".png", ".jpg", ".jpeg"} and is_chess_question:
    route = "chess_solver"

elif contains_youtube_url and asks_visual_count:
    route = "youtube_visual_solver"

elif contains_youtube_url and asks_spoken_words:
    route = "youtube_speech_solver"

elif wikipedia and temporal_constraint:
    route = "wikipedia_history_solver"

elif asks_table_argmax_min_or_operation:
    route = "structured_table_solver"

elif scholarly_paper_chain:
    route = "scholarly_solver"

else:
    route = "deep_research_fallback"
```

LLM classification may assist, but hard signals should dominate.

---

# 10. Web research subsystem

A maximum-score agent should have at least two independent search paths.

## 10.1 Search providers

Recommended configurable set:

```text
SEARCH_PRIMARY = Serper/Google or Tavily
SEARCH_SECONDARY = Bing/DDG/another independent backend
```

The point is not brand preference. It is resilience and search-result diversity.

Requirements:

- quoted search
- site-restricted search
- date constraints where supported
- multiple query reformulations
- language-specific queries
- source-domain preference
- result caching
- per-provider rate-limit handling

The implemented provider-neutral policy represents reformulations as an ordered
`SearchPlan`, validates that primary and secondary adapters declare different
backend families, and routes exhausted transient failures to the independent
secondary path. It applies bounded exponential retry with jitter and retry-after,
separate provider spacing, and a replaceable TTL cache keyed by provider plus the
complete request. Invalid queries and malformed provider output fail closed
instead of being hidden by fallback. The returned answer-free attempt trail does
not promote discovery snippets to evidence.

The provider-neutral boundary uses typed search requests/results and validates
provider name, exact query, contiguous ranks, result limits, unique URLs, and
aware retrieval timestamps before results enter research state. Provider runtime
failures remain distinct from malformed provider data so fallback policy can make
the correct decision.

---

## 10.2 Page retrieval ladder

```text
1. direct HTTP fetch
2. article/text extraction (trafilatura / BeautifulSoup)
3. alternate reader representation
4. JS browser (Playwright) if truly necessary
5. cached/archived version
6. alternate source or scholarly copy
```

Return structured data:

```json
{
  "url": "...",
  "status": 200,
  "title": "...",
  "published_at": "...",
  "retrieved_at": "...",
  "text": "...",
  "links": [...]
}
```

The retrieved-document boundary also binds the requested/final URLs and
extraction method to immutable raw/text artifact hashes. Large page text remains
in the artifact store instead of graph state.

The implemented C03 direct strategy streams within a byte ceiling, validates
redirect targets, classifies HTTP/block/JavaScript/incomplete-content failures,
and deterministically extracts visible HTML text, metadata, and links while
excluding navigation and active content. The coordinator routes incomplete or
JavaScript pages through reader then browser adapters, block pages through reader
then archive, and 404/410 directly to archive. Every strategy retains bounded
transient retries and C01 provenance validation; invalid requests and malformed
provider output fail closed. Raw and normalized text remain separate immutable
artifacts, and the safe attempt trail contains no response bodies.

Do not give the model raw navigation chrome if avoidable.

---

## 10.3 Source hierarchy

Prefer:

1. primary source directly referenced by the question
2. official/first-party source
3. authoritative database
4. scholarly paper
5. reputable secondary source
6. Wikipedia
7. search snippets

Search snippets are discovery evidence, not ideal final evidence.

---

# 11. Historical web and Wikipedia subsystem

This is a P0 feature for the current set.

## 11.1 MediaWiki API capabilities

Implement utilities for:

- page search
- page text
- revision IDs/timestamps
- latest revision at or before a cutoff
- old revision content
- links
- categories
- page history
- nomination/Featured Article process pages where applicable

Example conceptual API workflow:

```text
resolve title
   |
get revisions with timestamp <= cutoff
   |
select max timestamp
   |
fetch that revision
   |
extract target section/table
```

Do not use today's Wikipedia page for a question explicitly asking for the latest version during 2022.

---

## 11.2 Wayback/archive support

Implement archive fallback for:

- dated article pages
- historical rosters
- textbook pages that have changed
- source pages with current content inconsistent with requested time

Store snapshot timestamp in `Evidence`.

---

## 11.3 Temporal validity checker

Every piece of evidence for a historical task should be tested:

```python
def valid_for_constraint(evidence, temporal_constraint) -> bool:
    ...
```

A source retrieved today is fine if it explicitly documents the requested historical state. A source whose *content state* is current and unqualified is not necessarily sufficient.

---

# 12. Scholarly / PDF subsystem

Current tasks require following an article to a paper and extracting acknowledgments, and finding specimen/depository information in a scientific work.

## Required tools

- PyMuPDF (`fitz`) primary text extraction
- `pypdf` fallback
- PDF page rendering for scanned/layout-sensitive pages
- OCR fallback only when text extraction fails
- exact text search
- nearby-context extraction

Optional metadata/search:

- Crossref
- Semantic Scholar
- DOI resolution
- publisher page
- institutional repositories
- Internet Archive

## Search strategy

For award/funding questions:

```text
paper title -> exact PDF/text
find: "acknowledg", "support", "grant", author initials, "NASA"
extract exact award identifier in same sentence/paragraph
verify author-to-award relation
```

For specimen-deposition questions:

```text
paper -> find:
"deposited"
"depository"
"collection"
"holotype"
"paratype"
museum/institution acronym
specimen IDs

then resolve institution -> city
```

Do not infer city merely from an author's affiliation unless the paper says the specimens were deposited at that institution.

---

# 13. YouTube speech solver

The current set contains an exact-dialogue question. Semantic summaries are dangerous.

## Required ladder

1. Parse video ID.
2. Fetch official/manual captions if available.
3. Fetch auto-generated captions if needed.
4. Search timestamped transcript for anchor phrase.
5. Extract ±5–15 seconds around anchor.
6. Independently transcribe that audio slice with a strong ASR model.
7. Compare transcript variants.
8. If wording differs, adjudicate using the audio segment and phonetic context.
9. Return only the requested spoken response.

## Key rule

For “what did X reply?”:

> Optimize for **verbatim wording**, not semantic equivalence.

Do not “improve grammar.”

---

# 14. YouTube visual solver

A transcript cannot answer “highest number of distinct bird species simultaneously on camera.”

## Required pipeline

```text
download or stream video
        |
coarse frame sample (e.g. 1 fps)
        |
detect/count candidate species
        |
identify candidate maxima windows
        |
dense resample around those windows (e.g. 4–10 fps)
        |
track visible individuals/species across adjacent frames
        |
second visual pass on top candidate windows
        |
maximum simultaneous distinct-species result
```

### Robustness features

- contact-sheet generation for batches
- preserve timestamps
- query a vision model for *species identity and visible presence*, not only count
- have second VLM independently inspect top candidates
- if models disagree, resample higher resolution and/or more frames
- avoid double-counting the same species represented by multiple birds
- distinguish image overlays/photos from live-camera subjects if relevant to the video

The transcript can provide context but should not determine a visual maximum.

---

# 15. Audio attachment solver

Current tasks contain both semantic extraction (ingredients) and numeric extraction (page numbers).

## Required transcription strategy

- high-quality ASR model
- timestamped segments
- language auto-detection unless known
- at least two decode passes when the answer depends on a short ambiguous phrase or numbers
- optional second ASR model/provider for disputed segments

### Ingredient mode

Pipeline:

```text
ASR
 -> segment recipe sections
 -> identify "filling" portion
 -> ingredient entity extraction
 -> remove quantities/measurements
 -> exclude crust/topping/etc. if question says filling only
 -> canonicalize names conservatively
 -> alphabetical sort
 -> serializer
```

Do not let a generic language model invent conventional recipe ingredients not present in the recording.

### Numeric-page mode

```text
ASR
 -> extract spoken numeric spans
 -> normalize number words
 -> context-filter only requested pages
 -> integer parse
 -> deduplicate if appropriate
 -> numeric sort ascending
 -> comma-space serializer
```

Numbers should be checked against the audio segment because ASR frequently confuses teens/tens or adjacent numbers.

---

# 16. Chess image solver

This deserves a fully deterministic verification chain.

## 16.1 Image-to-board

Use two or more independent transcriptions:

- VLM A: describe every square/piece
- VLM B: independently produce FEN
- optional chessboard detector/OpenCV segmentation if useful

Then reconcile.

## 16.2 Validate FEN

Use `python-chess`:

- exactly one king each
- side-to-move matches question
- legal/possible position sanity
- piece placements match visual
- candidate move is legal

## 16.3 Engine solve

Use Stockfish with a meaningful depth/time budget.

For each candidate winning move:

- evaluate best move
- compare alternatives
- verify tactical result
- if <=7 pieces and applicable, use Syzygy tablebases for exactness

## 16.4 Serialize

Convert UCI move to SAN using the validated board.

Do not rely on a VLM to produce correct SAN from memory.

---

# 17. Python attachment solver

## Requirements

Execute attached Python in a sandbox:

- isolated subprocess/container
- network disabled unless explicitly needed
- CPU timeout
- memory limit
- restricted working directory
- capture stdout/stderr
- do not import local secrets
- avoid running arbitrary shell commands outside sandbox

Then:

1. inspect source statically
2. run program
3. identify what “final output” means from question
4. compare execution result with a quick independent/static calculation when feasible

For simple deterministic programs, the interpreter is authoritative.

---

# 18. Spreadsheet solver

The current XLSX task is high-value and entirely amenable to deterministic computation.

## Required process

1. Open with `openpyxl` and inspect:
   - sheet names
   - dimensions
   - headers
   - formulas
   - number formats
2. Load with pandas (`sheet_name=None`) for tabular analysis.
3. Detect formula cells and whether cached values exist.
4. If formulas matter and cached values are absent/stale, recompute via:
   - explicit formula logic when simple, or
   - a headless spreadsheet engine if needed.
5. Identify rows classified as food versus drinks from actual workbook semantics.
6. Use `Decimal`, not binary float, for currency totals.
7. Compute total twice:
   - pandas route
   - openpyxl/raw-cell route
8. Assert parity.
9. Format exactly to requested decimal places/currency style.

### Critical output rule

Do not automatically add `$` merely because the semantic value is money. Follow the question's exact requested output convention and the benchmark formatting guidance.

---

# 19. Structured table / mathematical solvers

## 19.1 Operation table

For a binary operation table:

```python
for a in elements:
    for b in elements:
        if op[a,b] != op[b,a]:
            involved.add(a)
            involved.add(b)
```

Then apply exact requested ordering.

This should never be solved by visual inspection if the table can be parsed.

## 19.2 Sports argmax/min

For “player with max X, return Y”:

```text
parse complete table
 -> numeric coercion
 -> assert no missing key rows
 -> calculate max/min
 -> assert tie handling
 -> select row
 -> retrieve requested field from same row
```

For tie rules:

```python
candidates = rows[value == min_value]
winner = sort_according_to_question(candidates)[0]
```

Do not trust a search snippet that already claims the winner without reproducing the reduction.

---

# 20. Cross-lingual/entity research

Current tasks include Polish and Japanese entity chains.

## Requirements

- search both English and native language
- normalize but preserve original spellings in evidence
- record stable entity identifiers where possible
- resolve namesakes by:
  - occupation
  - date
  - show/team
  - role
- when output requests Roman characters, use source-backed Romanization/transliteration rather than an arbitrary model transliteration
- when output requests only first/surname, strip other components deterministically after the entity is securely resolved

---

# 21. Deep-research fallback agent

Use a strong tool-using research agent only when a specialist solver does not fully resolve a task.

Hugging Face's Open Deep Research implementation provides a useful pattern:

- dedicated web/search agent
- webpage visiting/navigation/search-within-page
- archive search
- manager/code agent
- periodic planning/replanning
- enough steps to recover from weak first searches
- final answer reformulation

For this Level-1 subset, a suggested cap is:

```text
research max steps: 12–20
manager/reasoning max steps: 8–12
replan interval: ~3–4
```

These are starting points, not sacred values.

## Research prompt principles

The agent should:

- decompose unknown facts
- prefer primary evidence
- reformulate unsuccessful queries
- use site/domain restrictions
- search native language when relevant
- verify dates/versions
- explicitly track unresolved claims
- use Python for calculations
- not guess after tool failure
- return an evidence-backed candidate, not final free-form prose

---

# 22. Verification ladder

For maximum score, verification is not one “critic” call. Use the cheapest strongest check appropriate to the task.

## Tier 0 — structural validation

Always:

- answer not empty
- output type valid
- requested item count if known
- ordering constraint satisfied
- decimal places correct
- SAN parseable if chess
- IOC code shape if requested
- no prose prefix
- no markdown

## Tier 1 — deterministic validation

Examples:

- reverse-text round trip
- operation-table exhaustive test
- Python execution replay
- spreadsheet recomputation
- sports argmax/min replay
- numerical sort
- `python-chess` legal move validation

## Tier 2 — source validity

Check:

- primary vs secondary
- correct entity
- correct date/version
- exact field requested
- quoted/extracted evidence actually supports candidate

## Tier 3 — independent retrieval

For fragile factual tasks:

- second search backend
- different query wording
- native-language source
- archive version
- primary paper rather than article summary

## Tier 4 — independent model verifier

Use a model from a **different family/provider** when possible.

Give it:

- question
- output contract
- concise evidence packet
- candidate
- known risk flags

Ask for structured:

```json
{
  "verdict": "APPROVE | REJECT | UNCERTAIN",
  "reason_codes": [],
  "missing_evidence": [],
  "corrected_semantic_value": null
}
```

Do not ask it for a vague “does this look right?”

## Tier 5 — adjudicator

If candidate paths disagree:

- expose both evidence sets
- prefer direct/primary/date-valid evidence
- use deterministic checks where possible
- request targeted new research to resolve the *specific* conflict
- do not use simple majority vote

---

# 23. Multi-model strategy

Current GAIA reliability research shows no single frontier system is perfect. Model rankings also change quickly and performance varies by modality.

Therefore model choice should be configuration-driven.

Suggested roles as of this research snapshot:

```text
PRIMARY_RESEARCH_MODEL
SECONDARY_VERIFIER_MODEL
ADJUDICATOR_MODEL
VISION_MODEL_A
VISION_MODEL_B
ASR_MODEL / ASR_PROVIDER
```

## 23.1 Current empirical direction

Princeton HAL's current GAIA evaluation lists, among the systems it tested, raw accuracy roughly around:

- Gemini 3.5 Flash: 79.2%
- Gemini 3.1 Pro: 76.2%
- Claude Opus 4.7: 73.3%
- Claude Opus 4.5: 68.5%
- GPT-5.5: 62.8%

These are **not** guaranteed rankings for the course's selected 20 questions. Treat them as evidence that:

1. current frontier model quality matters
2. model families have different failure modes
3. no one model should be trusted as the only verifier for a 20/20 attempt

Before final deployment, run a clean regression suite on **non-target** GAIA/dev tasks or synthetic equivalents to compare accessible models.

## 23.2 Diversity beats repeated sampling

For a disputed answer, prefer:

```text
model A + source path A
versus
model B + source path B
```

over:

```text
same model at temperature 0.7 three times
```

Provider/model diversity can reduce correlated hallucinations.

## 23.3 Do not use model votes where a deterministic authority exists

Examples:

```text
chess       -> Stockfish
Python      -> Python interpreter
spreadsheet -> exact arithmetic
table       -> exhaustive computation
sorting     -> Python
```

---

# 24. Exact-match output architecture

This is a P0 system, not a cleanup regex.

## 24.1 Separate semantic answer from serialized answer

Never pass a raw LLM string directly to `/submit`.

Store:

```json
{
  "semantic_value": ["..."],
  "contract": {
    "answer_type": "list",
    "sort": "alphabetical",
    "delimiter": ", "
  },
  "serialized": "..."
}
```

## 24.2 Question-aware serializer

Examples:

### Integer

```python
return str(int(value))
```

### Decimal/currency

Use `Decimal`.

```python
quantized = value.quantize(Decimal("0.01"))
return f"{quantized:.2f}"
```

Only add currency symbols if explicitly required by the contract.

### Alphabetical list

Sort with a well-defined case/diacritic policy determined from the question. Default should preserve source spellings.

```python
items = sorted(items, key=...)
return ", ".join(items)
```

### Numeric list

```python
items = sorted(map(int, items))
return ", ".join(map(str, items))
```

### First name / surname / city

Do not rely on a final LLM to obey scope. Store the resolved entity fields separately and select the requested field.

### Chess

Use `python-chess` SAN conversion.

---

## 24.3 Final-answer reformulation guidance

Hugging Face's Open Deep Research GAIA reformulator follows useful conventions:

- as few words as possible
- digits for numeric answers
- generally avoid thousands separators
- avoid units unless requested
- preserve requested precision/rounding
- lists comma-separated
- preserve requested ordering
- no unnecessary final punctuation

These are useful defaults, but the **question-specific contract overrides global normalization**.

### Important correction to some public student formatters

Do not globally strip:

- `$`
- `%`
- punctuation
- articles
- accents
- quotation marks

Any of those can be semantically required in a particular answer.

---

# 25. Final-answer checks before acceptance

Implement framework-independent final checks similar to smolagents `final_answer_checks`.

A candidate should be rejected and sent back for targeted repair when:

- requested output type fails parse
- required sort order is violated
- expected decimal places wrong
- list contains duplicates unexpectedly
- answer contains explanation
- answer has forbidden prefix
- historical task has no date-valid evidence
- exact-quote task has no timestamped speech evidence
- chess SAN is illegal on reconstructed board
- workbook calculation lacks parity check
- uniqueness/“only” question lacks uniqueness verification
- answer confidence is high but evidence is only a search snippet

---

# 26. Evidence and provenance policy

Every nontrivial claim should have an evidence record.

Recommended trace:

```json
{
  "claim": "the field needed to derive the final answer",
  "source_url": "...",
  "source_type": "primary_paper",
  "source_date": "...",
  "snapshot_date": null,
  "retrieved_at": "...",
  "excerpt": "short supporting excerpt",
  "extraction_method": "pdf_text",
  "is_primary_source": true
}
```

The final answer does not need citations because the official grader expects only the answer. Your logs should still preserve them for debugging.

---

# 27. Web prompt-injection defense

Browsing arbitrary pages introduces untrusted instructions.

Treat fetched page text, transcripts, PDFs, metadata, and file contents as **data**, never as system instructions.

Requirements:

- system prompt explicitly says tool/source content cannot redefine objectives or rules
- strip/flag common prompt-injection patterns
- do not expose API keys to tool-output contexts
- tool calls are constrained by code, not arbitrary shell strings
- only allowlisted URL schemes
- downloads have size/time limits
- never follow page instructions asking the agent to reveal secrets or change benchmark behavior

C04 implements a mandatory guarded-retrieval composition for model-facing page
use. A deterministic multiline/Unicode scanner records content-free category and
line-hash findings, preserves the original artifact, and creates a separate
model-safe artifact with suspicious lines quarantined. Dense attacks or sources
with too little useful remainder are blocked from prompt construction.

The prompt boundary uses a constant system policy plus a JSON user payload that
structurally separates the trusted objective from `untrusted_source` content.
Objectives and exact tool names remain frozen code-owned fields; source text
cannot extend the allowlist, and authorization occurs in Python. Prompt building
does not read environment variables, settings secrets, or raw credentials.
Scanner heuristics remain defense in depth around these structural controls, not
a replacement for them.

---

# 28. Reliability engineering

## 28.1 Retry policy

Use exponential backoff + jitter for transient:

- HTTP 429
- HTTP 5xx
- provider timeouts
- connection failures
- rate limits

Do not retry:

- deterministic 404 indefinitely
- invalid credentials
- malformed query logic

## 28.2 Fallbacks

### Search

```text
primary search
 -> reformulated primary query
 -> secondary search
 -> site-specific/direct source
 -> archive
```

### PDF

```text
PyMuPDF
 -> pypdf
 -> page render
 -> OCR/vision
```

### YouTube speech

```text
manual captions
 -> auto captions
 -> audio ASR
```

### Attachment

```text
official /files
 -> official gated GAIA dataset
 -> fail loudly with metadata
```

### Model

```text
primary
 -> same-provider retry
 -> secondary provider
 -> adjudicator
```

---

## 28.3 Concurrency

Correctness is more important than throughput.

Use provider-specific semaphores:

```text
web requests: configurable
LLM provider A: 2–4 concurrent initially
LLM provider B: 2–4
video/ASR: 1–2
```

Measure actual quota behavior.

Hugging Face's Open Deep Research evaluation uses concurrent processing/checkpointing, but copying its exact concurrency without accounting for your APIs can produce rate-limit failures.

---

# 29. Checkpointing and observability

Public high-performing implementations repeatedly emphasize tracing/checkpoints.

Per task, persist:

```json
{
  "task_id": "...",
  "question_hash": "...",
  "attachment_sha256": "...",
  "task_profile": {...},
  "route": "...",
  "tool_events": [...],
  "sources": [...],
  "candidate_answers": [...],
  "verification_results": [...],
  "final_semantic_value": ...,
  "final_serialized_answer": "...",
  "errors": [...],
  "retries": {...},
  "timing": {...},
  "cost": {...},
  "software_version": "...",
  "model_versions": {...}
}
```

Use JSONL and/or SQLite.

## Resume behavior

A process restart should not throw away solved tasks.

Checkpoint after every accepted answer.

Allow:

```text
--resume
--rerun-failed
--rerun-uncertain
--task-id ...
--dry-run
--no-submit
```

---

# 30. Confidence should be evidence-based

Do not ask a model “How confident are you, 0–100?” and treat that as truth.

Compute a confidence/risk score from features:

```text
+ deterministic computation verified
+ primary source
+ correct historical snapshot
+ two independent sources agree
+ independent model agrees
- source inaccessible
- only snippet evidence
- ambiguous entity
- VLM-only numeric count
- ASR disagreement
- current source used for historical task
- unresolved tie/uniqueness condition
```

Use the score to decide when to spend more compute.

---

# 31. Adaptive compute budget

Spend extra work only where it reduces a known risk.

## Low-risk deterministic tasks

Examples:

- reverse string
- operation table
- Python execution

Policy:

```text
one solver
+ deterministic validation
+ no expensive model ensemble unless checks fail
```

## Medium-risk structured web tasks

```text
primary source extraction
+ deterministic reduction
+ secondary source check
```

## High-risk tasks

Examples:

- visual video maximum
- chess-board transcription
- exact audio phrase
- historical web snapshot
- obscure scholarly chain
- cross-lingual dated roster

Policy:

```text
multiple retrieval/extraction passes
+ independent verifier
+ targeted adjudication
```

---

# 32. Current-task-specific playbooks

These are implementation strategies only; they contain no benchmark answers.

## Task 1 — historical Wikipedia discography

1. Resolve exact English Wikipedia title.
2. Via MediaWiki API, choose latest revision timestamp <= 2022-12-31 23:59:59 UTC.
3. Parse discography/studio-album section.
4. Extract release years.
5. Count studio albums within 2000–2009 inclusive.
6. Check whether reissues/live/compilations are excluded.
7. Cross-check section against an archived or revision-rendered view.
8. Return integer.

Main traps:
- using current revision
- counting compilations/live albums
- off-by-one date range

---

## Task 2 — YouTube visual bird maximum

1. Obtain video.
2. Sample coarse frames.
3. Produce frame IDs/timestamps and candidate species sets.
4. Find top-count windows.
5. Dense-sample those intervals.
6. Inspect high-resolution crops if species are small.
7. Confirm simultaneous visibility on the *same frame/time*, not cumulative species in a scene.
8. Second independent vision pass on candidate maxima.
9. Return integer.

Main traps:
- using transcript
- accumulating species over time
- counting multiple individuals as multiple species
- missing short overlap window

---

## Task 3 — transformed/reversed text

1. Apply exact deterministic transform.
2. Round-trip to confirm original.
3. Preserve punctuation/case as encoded by transform unless question specifies otherwise.
4. Return exact resulting string.

---

## Task 4 — chess PNG

1. Download image.
2. Transcribe square occupancy twice.
3. Reconcile FEN.
4. Validate FEN.
5. Confirm black to move.
6. Run Stockfish.
7. Evaluate best move sufficiently deeply.
8. Confirm move legal and winning.
9. Convert to SAN.
10. Compare SAN to question notation request.
11. Return SAN only.

Main trap:
- confident VLM move without board/engine verification.

---

## Task 5 — November 2016 Featured Article nomination

1. Locate English Wikipedia Featured Article promotion/log data for November 2016.
2. Determine candidates promoted that month.
3. Filter to dinosaur article(s).
4. Assert uniqueness.
5. Open relevant FAC nomination page.
6. Extract nominator(s), distinguishing nominators from reviewers/supporters.
7. Cross-check page history/log.
8. Serialize names exactly as requested.

---

## Task 6 — noncommutative operation table

1. Parse elements/order.
2. Build matrix.
3. For all ordered/unordered pairs compare `a*b` and `b*a`.
4. Collect every element that occurs in at least one counterexample.
5. Assert exhaustive coverage.
6. Alphabetize exactly.
7. Join with comma-space.

---

## Task 7 — exact YouTube reply

1. Retrieve captions.
2. Find anchor “Isn't that hot?” allowing punctuation/caption variants.
3. Identify immediately following utterance and speaker.
4. Extract timestamp.
5. Independently ASR a short segment.
6. If transcript differs, listen/ASR variants and resolve exact wording.
7. Do not paraphrase.
8. Serialize reply only.

---

## Task 8 — historical LibreTexts veterinarian

1. Identify exact chemistry text/section.
2. Honor compiled date 2023-08-21.
3. Search within site and historical versions.
4. Locate equine-veterinarian reference in target context.
5. Resolve person's full identity from source.
6. Cross-check date/version.
7. Select surname only.

---

## Task 9 — botanical classification

For each listed item:
1. Identify botanical organ/category.
2. Explicitly evaluate whether it is botanically a fruit.
3. Keep only vegetables according to the question's exclusion rule.
4. Use trusted references for ambiguous cases.
5. Alphabetize requested result.
6. Preserve item spellings from prompt where possible.

Avoid culinary-only classification.

---

## Task 10 — strawberry-pie audio

1. Transcribe with timestamps.
2. Identify filling recipe section boundaries.
3. Extract all ingredient mentions in that section.
4. Exclude quantities/units.
5. Exclude crust ingredients.
6. Normalize ingredient names conservatively.
7. Second-pass ASR disputed words.
8. Alphabetize.
9. Return comma-separated ingredients.

---

## Task 11 — Polish TV entity chain

1. Identify Polish-language adaptation/version and actor.
2. Confirm actor identity from a Polish/native or authoritative source.
3. Search that actor's role in `Magda M.`.
4. Cross-check with cast source.
5. Extract requested first name only.

Main traps:
- actor/character confusion
- namesake actor
- translating a character name unnecessarily

---

## Task 12 — Python source

1. Read source.
2. Inspect for external I/O/random/time dependencies.
3. Sandbox execute.
4. Capture final printed/result value.
5. If deterministic/simple, independently derive or rerun in fresh process.
6. Return numeric/string form requested.

---

## Task 13 — 1977 Yankees walks/at-bats

1. Fetch authoritative 1977 Yankees regular-season batting stats.
2. Ensure table is regular season, not postseason/career.
3. Parse BB and AB columns numerically.
4. Compute maximum BB.
5. Assert tie status.
6. Retrieve AB from the same row.
7. Secondary stats source cross-check.
8. Return integer.

---

## Task 14 — homework audio page numbers

1. Transcribe.
2. Extract page-number phrases only.
3. Convert number words to integers.
4. Check ambiguous tens/teens against audio.
5. Sort numeric ascending.
6. Deduplicate only if repeated instruction is clearly same page.
7. Join comma-space.

---

## Task 15 — Universe Today → paper → NASA award

1. Find the exact June 6, 2023 Universe Today article specified.
2. Identify the actual linked research paper.
3. Fetch paper primary text/PDF.
4. Locate acknowledgments/funding.
5. Search named author initials/name.
6. Determine which NASA award explicitly supported that author/work.
7. Cross-check award spelling/identifier.
8. Return exact award identifier, not an agency/program generalization.

---

## Task 16 — Nedoshivina 2010 specimen deposition

1. Resolve exact paper.
2. Obtain full PDF/text.
3. Search taxon/specimen sections and repository statements.
4. Identify the eventual depository/institution of the Vietnamese specimens described by Kuznetzov.
5. Verify institution expansion/acronym.
6. Resolve institution's city from authoritative source.
7. Return city only.

---

## Task 17 — 1928 Olympics least athletes + IOC code

1. Get country/delegation counts for 1928 Summer Olympics.
2. Parse all relevant participating nations.
3. Compute minimum athlete count.
4. Collect all ties.
5. Alphabetically choose country if question says tie-break that way.
6. Map historical country name to requested IOC code using appropriate Olympic nomenclature.
7. Cross-check with a second Olympic/reference source.
8. Return code only.

---

## Task 18 — Japanese baseball jersey-number neighbors

1. Resolve Taishō Tamai's team and jersey number **as of July 2023**.
2. Get roster/jersey-number source valid for that date.
3. Calculate immediate numeric neighbors `n-1`, `n+1`.
4. Find players assigned those numbers at that date.
5. Verify both requested players satisfy “pitcher” condition.
6. Resolve Roman-character surnames from official/authoritative profile.
7. Cross-check historical roster snapshot.
8. Return requested surnames in required order.

Main traps:
- current roster
- number reassignment
- Japanese romanization variants
- player position changed/misread

---

## Task 19 — Excel food sales

1. Resolve attachment robustly.
2. Inspect all sheets.
3. Identify schema and sales columns.
4. Identify food vs drink categories based on workbook content/question semantics.
5. Sum food sales with `Decimal`.
6. Recompute via second independent parser.
7. Assert no hidden/total rows double-counted.
8. Format exactly two decimals and currency convention requested.

---

## Task 20 — Malko Competition historical nationality

1. Build recipient/winner records after 1977.
2. Extract nationality at the relevant competition/biographical context.
3. Determine which nationality corresponds to a state/country that no longer exists.
4. Assert exactly one matching recipient under the question wording.
5. Verify nationality from a second source.
6. Extract first name only.
7. Serialize exact source-backed spelling.

---

# 33. Benchmark hygiene / anti-leak policy

The GAIA dataset is gated partly to reduce benchmark contamination and asks users not to redistribute validation/test material in crawlable form.

For a legitimate public course project:

## Do

- use the public `/questions` prompts
- use the official gated GAIA dataset to obtain attachments when the official file endpoint is broken
- train/test your architecture on separate GAIA/dev tasks or synthetic tasks
- maintain a public, inspectable implementation
- collect public primary-source evidence

## Do not

- query the web for `<task_id> answer`
- search GitHub specifically for hard-coded answers to current task IDs
- copy gold answers from leaked repositories
- hard-code known target outputs
- expose gated `Final answer` fields in logs/public repo
- tune against the target gold answers until exact match

This matters both ethically and technically: a solution built around leaks proves nothing and may be reviewed on a leaderboard that explicitly warns high scores can be checked.

---

# 34. Testing strategy without target leakage

## 34.1 Unit tests

Every tool should have deterministic tests:

- output serializer
- ordering
- currency/Decimal
- MediaWiki revision selection
- table argmax/min
- operation table checker
- attachment MIME validation
- FEN/SAN conversion
- transcript anchor matching
- retry/backoff logic
- submission preflight

For deterministic solver invariants, combine generated property tests with small
poisoned implementations. Required mutation families include off-by-one scans,
row-order tie selection, ignored output sorting, binary-float currency totals,
and incomplete table acceptance. Use only synthetic or permitted non-target data.

## 34.2 Modality fixtures

Create synthetic/local fixtures:

- PNG chess positions with known engine moves
- MP3/TTS clips with known ingredient/page-number content
- XLSX food/drink tables with known totals
- Python scripts with known outputs
- mock historical wiki revisions
- mock YouTube transcript snippets

## 34.3 Clean GAIA regression

Use non-target development/held-out tasks where permitted.

Track metrics by class:

```text
web factual
historical web
PDF/scholarly
spreadsheet
audio
vision
video
code
table math
exact formatting
```

Do not optimize only aggregate score; find brittle modalities.

---

# 35. Evaluation metrics for your own system

Track:

- semantic correctness on clean dev set
- exact-match correctness
- tool success rate
- attachment retrieval success
- source-primary rate
- historical-date-valid evidence rate
- verifier rejection rate
- retry recovery rate
- answer-contract failure rate
- cost/question
- wall-clock/question
- model/tool error distribution

For maximum score, an exact-format failure rate even of 5% is unacceptable.

---

# 36. Pre-submission risk dashboard

Before official submission, print:

```text
Task   Route               Candidate   Evidence   Deterministic checks   Risk
01     historical_wiki     READY       2 primary PASS                   LOW
02     youtube_visual      READY       frames    PASS+2 VLM             MED
...
20     historical_entity   READY       3 sources PASS                   LOW
```

Submission should require:

- 20/20 candidate answers present
- 20 unique IDs
- no `UNRESOLVED`
- no unhandled attachment
- no failed output contracts
- no historical question backed only by a current undated source
- no chess answer without engine verification
- no audio exact-quote answer without timestamp evidence
- no spreadsheet answer without deterministic recomputation

For remaining medium/high-risk tasks, spend extra compute/research before submission.

---

# 37. Recommended project structure

```text
gaia-max/
├── app.py
├── pyproject.toml
├── README.md
├── .env.example
│
├── src/gaia_max/
│   ├── config.py
│   ├── state.py
│   ├── contracts.py
│   ├── task_analyzer.py
│   ├── router.py
│   ├── graph.py
│   ├── runner.py
│   ├── submission.py
│   ├── checkpoint.py
│   ├── evidence.py
│   │
│   ├── clients/
│   │   ├── gaia_api.py
│   │   ├── hf_dataset.py
│   │   ├── models.py
│   │   └── search.py
│   │
│   ├── retrieval/
│   │   ├── web.py
│   │   ├── wikipedia.py
│   │   ├── archive.py
│   │   ├── scholarly.py
│   │   └── youtube.py
│   │
│   ├── media/
│   │   ├── audio.py
│   │   ├── video.py
│   │   └── vision.py
│   │
│   ├── solvers/
│   │   ├── deterministic_text.py
│   │   ├── structured_table.py
│   │   ├── code.py
│   │   ├── spreadsheet.py
│   │   ├── audio.py
│   │   ├── youtube_speech.py
│   │   ├── youtube_visual.py
│   │   ├── chess.py
│   │   ├── wikipedia_history.py
│   │   ├── scholarly.py
│   │   ├── crosslingual.py
│   │   └── deep_research.py
│   │
│   ├── verification/
│   │   ├── structural.py
│   │   ├── deterministic.py
│   │   ├── source_validity.py
│   │   ├── independent_retrieval.py
│   │   ├── critic.py
│   │   └── adjudicator.py
│   │
│   └── formatting/
│       ├── contract_parser.py
│       ├── serializer.py
│       └── final_checks.py
│
├── tests/
│   ├── unit/
│   ├── fixtures/
│   ├── integration/
│   └── regression/
│
├── scripts/
│   ├── snapshot_questions.py
│   ├── dry_run.py
│   ├── inspect_task.py
│   ├── compare_models.py
│   └── submit.py
│
└── runs/
    ├── checkpoints/
    ├── traces/
    └── evidence_cache/
```

---

# 38. Suggested dependencies

Pin versions after integration testing.

Core:

```text
requests / httpx
pydantic
tenacity
python-dotenv
langgraph (if using LangGraph)
```

Research:

```text
tavily-python and/or Google/Serper client
beautifulsoup4
trafilatura
playwright (fallback)
internetarchive / Wayback client if desired
```

Wikipedia:

```text
requests
mwparserfromhell
```

Data:

```text
pandas
openpyxl
pyarrow
```

Documents:

```text
pymupdf
pypdf
python-docx
```

Media:

```text
ffmpeg
yt-dlp
youtube-transcript-api
faster-whisper or equivalent strong ASR
Pillow
opencv-python-headless
```

Chess:

```text
python-chess
Stockfish executable
```

HF:

```text
huggingface_hub
datasets
```

Testing/quality:

```text
pytest
pytest-asyncio
ruff
mypy or pyright
```

Do not install every library merely because it appears here. Use lockfiles and test the final Space environment.

---

# 39. Configuration requirements

Example:

```text
GAIA_API_URL=https://agents-course-unit4-scoring.hf.space
HF_TOKEN=...

PRIMARY_MODEL_PROVIDER=...
PRIMARY_MODEL=...

SECONDARY_MODEL_PROVIDER=...
SECONDARY_MODEL=...

ADJUDICATOR_MODEL_PROVIDER=...
ADJUDICATOR_MODEL=...

VISION_MODEL_A_PROVIDER=...
VISION_MODEL_A=...
VISION_MODEL_B_PROVIDER=...
VISION_MODEL_B=...

ASR_BACKEND=...
ASR_MODEL=...

SEARCH_PRIMARY=...
SEARCH_SECONDARY=...

MAX_RESEARCH_STEPS=16
MAX_REPLAN=3
MAX_VERIFIER_RETRIES=3

HTTP_TIMEOUT=30
DOWNLOAD_TIMEOUT=120

DRY_RUN=true
ALLOW_SUBMIT=false
```

Official submission should require an explicit action/config toggle so a development run cannot accidentally submit.

---

# 40. Prompt architecture

Do not use one enormous prompt for every role.

## 40.1 Planner/researcher

Responsibilities:

- identify unknowns
- choose evidence path
- respect temporal constraints
- use tools
- expose concise evidence/candidate
- never decide serialization details beyond semantic answer

## 40.2 Tool executor

Tool inputs should be narrow and structured.

Bad:

```text
"Figure everything out"
```

Good:

```json
{
  "query": "\"exact paper title\" NASA acknowledgments R.G. Arendt",
  "preferred_domains": ["publisher", "nasa.gov"]
}
```

## 40.3 Verifier

Should receive:

- exact question
- parsed contract
- candidate semantic value
- evidence packet
- deterministic check results
- known risk flags

It should attack the candidate, not merely restate it.

## 40.4 Formatter

Prefer deterministic code.

If an LLM is used only to infer a complex output contract, validate the contract with code before serialization.

---

# 41. Lessons extracted from public implementations

## 41.1 Hugging Face Open Deep Research / smolagents

Valuable ideas:

- dedicated web agent managed by a higher-level reasoning/code agent
- planning intervals instead of endless reactive drift
- webpage navigation/find/archive capabilities
- code execution as a first-class reasoning tool
- image conversion for awkward PDF/XLS visual cases
- JSONL evaluation/checkpointing
- dedicated final-answer reformulator

Reported full GAIA validation performance is far above basic agents, but still not perfect; therefore copy the architectural lessons, not the assumption that one research loop solves all modalities.

---

## 41.2 `yc1838/huggingface-agents-course-final`

This is one of the most useful inspectable high-scoring student architectures found in research. The student leaderboard showed a reported 95 score, though the course itself warns leaderboard scores are not necessarily fully verified.

Useful design choices:

```text
Perception
 -> Planner
 -> rule-based Router
 -> cheap/strong Executors
 -> Verifier
 -> on rejection, re-plan
 -> Formatter
```

Especially useful:

- rule-based routing
- strong model for planning/verification
- checkpoints
- provider configuration
- Web/Python/file/audio/YouTube tools
- verifier loops back to planner rather than merely replaying same execution

Improvements for a 20/20-oriented implementation:

- replace generic file handling with richer modality-specific solvers
- add official gated-dataset attachment fallback
- add historical Wikipedia/Wayback logic
- add true YouTube visual analysis
- add Stockfish chess verification
- make formatter contract-aware rather than globally stripping punctuation
- use independent-provider verification on high-risk tasks

---

## 41.3 `SpaceFozzy/gaia-agent`

Reported:

- 11/20 before external file support
- later 16/20 after improvements including expanded file support
- LangGraph
- Claude
- Tavily/math
- document/audio support
- MLflow tracing

Lesson:

> File/modality support can move the score dramatically. Observability makes iteration possible.

---

## 41.4 `tqv-notes/gaia_agent`

Reported 75% and uses a practical tool stack:

- strong Claude model
- Tavily
- DuckDuckGo fallback
- Wikipedia
- YouTube transcripts
- Python
- Whisper
- PDF
- pandas
- image vision

Important anecdote: the author reports at least one miss that appears to be pure output formatting (semantic money value correct, formatted with currency/thousands punctuation differently from ground truth).

Lesson:

> Exact formatting is a score feature, not UI polish.

---

## 41.5 `e-alizadeh/GAIA-Agent`

Useful pattern:

```text
route_question
 -> invoke_tools
 -> synthesize_response
 -> format_output
```

with Tavily/DDG fallback plus media/file tools.

Lesson:

> Separate routing/tool use/output formatting even in simpler architectures.

---

# 42. What not to copy blindly from public agents

- leaderboard score alone — official course says scores may be unverified
- leaked/hard-coded target answers
- giant single prompt
- LLM-only arithmetic
- VLM-only chess
- transcript-only video vision
- current Wikipedia for date-specific historical questions
- one search provider
- one ASR transcript for ambiguous numbers/quotes
- global answer regex that strips potentially meaningful punctuation
- `/files/{task_id}` as the sole attachment source
- “confidence = model says 95%”
- retries that repeat the exact same failed path

---

# 43. Implementation priority for a full-score attempt

## P0 — mandatory

1. Question snapshot/integrity check
2. Robust attachment resolver with gated-dataset fallback
3. Output-contract parser + deterministic serializer
4. Search + page retrieval + second search fallback
5. Historical Wikipedia/revision support
6. Python execution
7. XLSX deterministic parser/calculator
8. High-quality audio transcription
9. YouTube captions + audio extraction
10. YouTube visual frame analysis
11. Chess FEN + `python-chess` + Stockfish
12. PDF/scholarly extraction
13. Deterministic table/math utilities
14. Date/source evidence tracking
15. Verification ladder
16. Per-task checkpointing
17. Submission preflight

## P1 — high value

- Wayback/archive integration
- bilingual/native-language query generator
- second VLM
- second ASR backend/model
- independent LLM verifier
- provider fallbacks
- browser/Playwright fallback
- automated risk scoring
- source caching
- model comparison harness

## P2 — useful but not primary score drivers

- polished UI
- elaborate dashboard
- broad RAG/vector DB
- long-term memory
- fine-tuning
- generic multi-agent social simulation

---

# 44. Suggested implementation sequence for Codex

The goal is to preserve correctness while adding capability incrementally.

### Milestone A — contract and infrastructure

Build:

- typed state models
- official API client
- question snapshot
- attachment resolver
- checkpoint store
- output contract + serializer
- dry-run runner
- submission preflight

Acceptance:

- fetches 20 current tasks
- detects 5 current attachments
- obtains all attachment bytes through primary/fallback path
- never calls `/submit` during dry-run
- can serialize test fixtures exactly

### Milestone B — deterministic solvers

Build:

- transformed-text
- operation-table
- Python
- spreadsheet
- structured min/max/tie
- sorting/name-field utilities

Acceptance:

- synthetic fixture suite 100%
- two independent spreadsheet calculations agree
- Python sandbox tests pass

### Milestone C — web/historical research

Build:

- primary/secondary search
- robust page fetch
- MediaWiki revisions
- archive
- scholarly PDF
- evidence model

Acceptance:

- synthetic/clean historical questions select correct dated revision
- primary-source trace preserved

### Milestone D — audio/video

Build:

- ASR with timestamps
- YouTube transcript
- yt-dlp/ffmpeg slice
- visual frame sampler/contact sheets
- vision-model interface
- disagreement handling

Acceptance:

- known clips exact transcript extraction
- numeric speech tests
- synthetic video max-count tests

### Milestone E — chess

Build:

- image transcription interface
- FEN reconciliation
- legal-state checks
- Stockfish
- SAN serialization

Acceptance:

- suite of known chess diagrams solved/validated

### Milestone F — ensemble verification

Build:

- structural verifier
- source/date verifier
- independent retrieval
- secondary-model critic
- adjudicator

Acceptance:

- intentionally poisoned candidates are rejected
- ambiguous evidence triggers targeted retry

### Milestone G — current evaluation dry run

Run all current prompts **without target gold answers**.

Review only:

- crashes
- missing evidence
- risk flags
- contradictory sources
- formatting contract failures
- attachment failures
- high-cost loops

Resolve every engineering failure.

### Milestone H — final deployment

- pin dependencies
- configure Secrets
- public Space
- clean rebuild
- run dry mode in Space
- validate 20-answer payload
- deliberate official submit

---

# 45. Definition of done for “best possible score”

A submission candidate is ready when:

```text
[ ] Current /questions snapshot has expected count and no unknown profile.
[ ] Every attachment task resolves successfully.
[ ] Every task has a specialized or well-tested fallback route.
[ ] Every final candidate has a parsed output contract.
[ ] Every deterministic task passes deterministic verification.
[ ] Every historical task has time-valid evidence.
[ ] Every scholarly task uses the primary paper when obtainable.
[ ] Every exact speech task has timestamped audio/caption evidence.
[ ] Video visual task has frame-based, not transcript-only, analysis.
[ ] Chess task has validated board + engine-derived legal SAN.
[ ] XLSX task has two independent recomputations.
[ ] High-risk web/entity tasks have an independent verification path.
[ ] No final answer contains prose/markdown/prefix.
[ ] Exactly one answer exists for each of the 20 current task IDs.
[ ] Checkpoints allow recovery from API/provider interruption.
[ ] Submission is gated behind explicit human action/config.
[ ] Public repository contains no secrets and no benchmark gold answers.
```

---

# 46. Research sources and what each contributes

## Official Hugging Face course

### Final assignment hands-on
https://huggingface.co/learn/agents-course/en/unit4/hands-on

Confirms:
- 20 Level-1 validation questions
- filtered by tools/steps
- API routes
- exact-match scoring
- answer-only requirement
- public code-link expectation
- warning that leaderboard can contain unverified scores

### What is GAIA?
https://huggingface.co/learn/agents-course/en/unit4/what-is-gaia

Explains:
- benchmark goals
- multimodal/tool/research nature
- difficulty levels

### Official course repository
https://github.com/huggingface/agents-course

Useful for current source-controlled course content.

---

## Official scoring implementation

### Scoring Space source
https://huggingface.co/spaces/agents-course/Unit4_scoring/blob/main/main.py

Critical facts:
- dataset config/split
- `<3` tools and `<6` steps filter
- exact comparison logic
- denominator
- submission handling
- current attachment path implementation
- leaderboard update behavior

### Current live questions
https://agents-course-unit4-scoring.hf.space/questions

Used only to derive the current task capability matrix; no hidden answers.

---

## GAIA dataset

https://huggingface.co/datasets/gaia-benchmark/GAIA

Important for:
- official attachment access
- dataset structure
- gated/anti-contamination policy
- development methodology

---

## Hugging Face Open Deep Research / smolagents

https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research

https://raw.githubusercontent.com/huggingface/smolagents/main/examples/open_deep_research/scripts/reformulator.py

Important for:
- web-agent + manager architecture
- planning intervals
- web tools/navigation/archive patterns
- file/media handling ideas
- GAIA answer reformulation conventions
- checkpoint/evaluation patterns

---

## Princeton HAL GAIA Reliability

https://hal.cs.princeton.edu/reliability/benchmark/gaia/

https://hal.cs.princeton.edu/reliability/benchmark/gaia/analysis/

Important for:
- current frontier-agent accuracy comparisons
- evidence that confidence != correctness
- modality/tool failure modes
- current chess-task example showing VLM-only reasoning failure
- rationale for independent verification and deterministic domain tools

---

## Public student implementations

### yc1838
https://github.com/yc1838/huggingface-agents-course-final

Useful:
- Plan → Execute → Verify
- rule-based routing
- model tiers
- checkpointing
- provider abstraction
- verifier/replanning
- inspectable high-score-oriented architecture

### SpaceFozzy
https://github.com/SpaceFozzy/gaia-agent

Useful:
- empirical improvement after richer file support
- LangGraph
- observability
- audio/document tools

### tqv-notes
https://github.com/tqv-notes/gaia_agent

Useful:
- broad practical tool stack
- search fallbacks
- Whisper/PDF/Excel/vision/YouTube
- reported formatting-related point loss

### e-alizadeh
https://github.com/e-alizadeh/GAIA-Agent

Useful:
- clean routing → tools → synthesis → format architecture
- search fallback and multimodal tool coverage

---

## Hugging Face community operational reports

### Attachment endpoint issue
https://discuss.huggingface.co/t/attachements-not-available-on-https-agents-course-unit4-scoring-hf-space-docs/171219

Important:
- current/2026 evidence that `/files/{task_id}` can fail
- exact current Excel attachment example
- reason to require official-dataset fallback

### Per-question score discussion
https://discuss.huggingface.co/t/agent-course-final-project-can-we-see-the-score-for-each-question/171760

Important:
- official `/submit` response is aggregate-only
- reason to build local instrumentation rather than treat official scoring as debugging telemetry

---

# 47. Final architectural recommendation

If starting from zero today, build:

```text
LangGraph/custom typed control plane
        |
        +-- deterministic task analyzer/router
        |
        +-- specialist solvers
        |     code
        |     xlsx
        |     audio
        |     youtube speech
        |     youtube vision
        |     chess
        |     historical wiki
        |     scholarly PDF
        |     structured tables
        |
        +-- deep-research fallback
        |     primary + secondary search
        |     page reader
        |     archives
        |     bilingual queries
        |
        +-- evidence store
        |
        +-- verification ladder
        |     deterministic
        |     date/source
        |     independent retrieval
        |     cross-model critic
        |     adjudicator
        |
        +-- question-specific output serializer
        |
        +-- checkpoints + dry runner
        |
        +-- explicit submission preflight
```

For a 30% pass, this is over-engineered.

For a serious **20/20 attempt**, the redundancy is the point.

The benchmark's current task set contains several questions for which domain-native verification can nearly eliminate an entire class of model error. The remaining difficult web/history/media questions should receive more retrieval diversity and independent verification rather than simply more chain-of-thought or more agent steps.

The system should aim to make the LLM the **research/reasoning component**, not the sole source of truth.

---

# 48. Codex work order — concise version

When coding begins, instruct Codex:

> Implement the system in this spec in milestones. Preserve an explicit distinction between semantic answer, evidence, verification, and serialized answer. Use deterministic specialist solvers before a generic research agent. Build the current attachment fallback first. Treat date/version constraints as first-class. Use Stockfish for chess, exact Python/Decimal for code and spreadsheets, timestamped ASR for audio, real video frames for visual YouTube, MediaWiki revisions/archives for historical pages, and primary-paper extraction for scholarly tasks. Add independent source/model verification only where deterministic checks cannot establish the result. Never hard-code target answers or inspect the target gold-answer field. Keep every tool trace/checkpoint reproducible and require a 20-unique-answer preflight before `/submit`.
