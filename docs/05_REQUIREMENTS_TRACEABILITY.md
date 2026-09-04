# Requirements Traceability

## Purpose

This matrix prevents the implementation backlog from drifting away from the source-of-truth specification. It maps every numbered section in `GAIA_MAX_SCORE_ENGINEERING_SPEC.md` to architecture documents and implementation stories.

When a requirement changes, update:

1. The source specification if appropriate
2. The affected design section
3. The affected story acceptance criteria
4. The tests that prove it

## Full section mapping

| Spec section | Requirement theme | Design location | Implementation stories |
|---:|---|---|---|
| 0 | Codex directive, priorities, anti-hard-coding | README boundaries; System Design §§1–2 | All stories; especially A06, A10, G01, H05 |
| 1 | Deterministic outer graph plus specialists and verification | Beginner Architecture §§1–5; System Design §§2–3 | A15–A18, B01–B07, F04–F09 |
| 2 | Official evaluation routes, exact match, denominator, IDs | System Design §§4, 13, 17 | A04–A05, A13–A14, A19, G05–G06, H03 |
| 3 | Broken attachment path and official gated fallback | System Design §§8, 19 | A07–A10, G02 |
| 4 | Current 20-task capability matrix | Beginner Architecture §7; answer-free profiles | A06, A17, B–F specialist stories |
| 5 | Generic ReAct failure modes | Beginner Architecture §§2–3 | A11–A17, F01–F09 |
| 6 | Recommended orchestration graph | Beginner Architecture §§3–4; System Design §§3–5 | A15–A18 |
| 7 | Typed core state/data models | System Design §§4–6 | A03, A15, A18 |
| 8 | Task analyzer and flags | System Design §§5–6 | A11–A12 |
| 9 | Deterministic router | System Design §5 | A06, A17 |
| 10 | Dual search and page retrieval | System Design §§7, 9 | C01–C04 |
| 11 | Historical Wikipedia, archives, temporal validity | System Design §10 | C05–C07 |
| 12 | Scholarly/PDF extraction | Beginner Architecture §7; System Design §11 | C08–C10 |
| 13 | YouTube speech ladder | Beginner Architecture §7 | D06–D07 |
| 14 | YouTube visual frame pipeline | Beginner Architecture §7 | D08–D11 |
| 15 | Audio ingredient and numeric modes | Beginner Architecture §7 | D01–D05 |
| 16 | Chess transcription, FEN, Stockfish, SAN | Beginner Architecture §§5, 7 | E01–E05 |
| 17 | Sandboxed Python execution | Beginner Architecture §§5, 7 | B04 |
| 18 | Spreadsheet dual computation and exact currency | Beginner Architecture §§5–6, 34 | B05–B08, A13 |
| 19 | Operation tables and sports reduction | Beginner Architecture §§5, 7 | B02–B03, B07–B08 |
| 20 | Cross-lingual/entity research | Beginner Architecture §7 | C11–C13, F06 |
| 21 | Bounded deep-research fallback | System Design §§2, 15 | F01–F03 |
| 22 | Verification ladder tiers 0–5 | System Design §11 | F04–F08 |
| 23 | Multi-model roles and diversity | System Design §§7, 11 | F06–F09; clean model comparison during G |
| 24 | Semantic answer versus exact serialization | Beginner Architecture §2; System Design §13 | A11, A13–A14 |
| 25 | Final answer checks | System Design §13 | A14, G05 |
| 26 | Evidence and provenance | System Design §§6, 8–9 | C13, F05–F06 |
| 27 | Web prompt-injection defense | System Design §16 | C04, F10, H05 |
| 28 | Retry, fallback, and concurrency | System Design §§4–5, 7 | A04, A09–A10, A18, C01–C03, F12 |
| 29 | Checkpointing, observability, and resume CLI | System Design §§4, 14 | A16, A18–A19, F10–F12 |
| 30 | Evidence-derived confidence | System Design §12 | F09, F11 |
| 31 | Adaptive compute budgets | System Design §§11–12 | F04–F09, F11 |
| 32 | Current-task-specific playbooks | Answer-free profiles and solver policies | A06, A17, all B–F solver stories |
| 33 | Benchmark hygiene and anti-leak policy | README boundaries; System Design §19 | A06, A10, all tests, G01, H05 |
| 34 | Testing without target leakage | Learning Workflow §2; Backlog milestone gates | B08, all acceptance tests, F12, G01 |
| 35 | Internal evaluation metrics | System Design §§12, 14 | F09–F11 |
| 36 | Pre-submission risk dashboard | System Design §§12, 17 | F11, G02–G06 |
| 37 | Project structure | System Design and repository layout in final design | A01 and all file-creation stories |
| 38 | Dependency set and pinning | System Design §18 | A01, H01, H04 |
| 39 | Environment configuration | System Design §§7, 18 | A02, H01–H04 |
| 40 | Separate prompts by role | System Design §15 | A12, D09, F01–F08 |
| 41 | Lessons from public implementations | Architecture choices ADR-001 through ADR-005 | A15–A18, F01–F12 |
| 42 | Patterns not to copy blindly | README boundaries; System Design ADRs | A13–A14, C04–C07, D07–D11, E02–E05, F06–F08 |
| 43 | P0/P1/P2 priorities | Backlog maximum-score execution policy | Current-task coverage and dry-run evidence determine order; optional capabilities remain behind core gates |
| 44 | Milestones A–H | Entire implementation backlog | A01–H07 |
| 45 | Definition of done | Backlog milestone gates and cross-cutting DoD | G01–G06, H01–H07 |
| 46 | Research sources | README official references | Documentation review; H05 public methodology |
| 47 | Final architectural recommendation | Beginner Architecture and System Design | A15–A18, specialist and verification epics |
| 48 | Concise Codex work order | Learning Workflow; prioritized backlog | Score-critical stories first; conditional/deferred stories are pulled forward only by dry-run evidence |

## Current-task route coverage

This table contains no answers. It verifies that every current public task shape has an implementation and verification path.

| Public task shape | Primary route | Required stories | Required verification |
|---|---|---|---|
| Historical discography count | `wikipedia_history` | C05–C07 | Dated revision + deterministic count |
| Visual bird maximum | `youtube_visual` | D06, D08–D11 | Full-frame coverage + independent visual pass |
| Transformed text | `deterministic_text` | B01, B07 | Round-trip invariant + replay match |
| Chess PNG | `chess` | E01–E05 | Dual board transcription + Stockfish + SAN |
| Featured Article nominator | `historical_web` | C02–C07, C12–C13 | Uniqueness + nomination page/log consistency |
| Noncommutative table | `operation_table` | B02, B07–B08 | Exhaustive pair check |
| Exact YouTube reply | `youtube_speech` | D02–D03, D06–D07 | Caption timestamp + independent ASR |
| Historical LibreTexts detail | `historical_web` | C02–C07 | Compiled-date validity + exact entity field |
| Botanical filtering | `structured_research` | C02, C12–C13 | Item-by-item classification + ambiguous references |
| Recipe MP3 | `audio_ingredient` | D01–D04 | Section boundary + second decode |
| Polish TV chain | `crosslingual` | C11–C13 | Native/entity confirmation |
| Python attachment | `code` | B04, B07 | Execution + static/replay check |
| Historical batting table | `structured_table` | B03, C12 | Full table reduction + tie/source check |
| Homework MP3 | `audio_numeric` | D01–D03, D05 | Numeric spans + disputed audio check |
| Article-to-paper award | `scholarly` | C02–C03, C08–C10 | Primary paper acknowledgment relationship |
| Specimen deposition | `scholarly` | C08–C10 | Depository passage + institution/city resolution |
| Olympics least athletes | `structured_table` | B03, C12 | Full min/tie reduction + IOC mapping |
| Dated Japanese roster | `crosslingual_historical` | C07, C11–C13 | Date-valid native roster + Romanization |
| XLSX food sales | `spreadsheet` | B05–B08 | Dual exact recomputation + Decimal mutation tests |
| Historical nationality | `structured_historical` | C07, C11–C13 | Uniqueness + independent nationality validation |

## Cross-cutting test traceability

| Risk | Required test class |
|---|---|
| Formatting causes exact-match loss | Serializer golden and negative tests |
| Question set changes | Snapshot mutation tests |
| Attachment endpoint fails | Primary-404-to-gated-fallback integration test |
| Gated answer leakage | Schema allowlist and log-capture security tests |
| Historical source is current | Temporal validity fixtures |
| Search snippet treated as proof | Evidence-policy negative tests |
| Model invents after source failure | `NOT_FOUND` and blocking-path tests |
| VLM gives unsupported chess move | Illegal FEN/move and engine-authority tests |
| Transcript used for visual question | Router and evidence-policy tests |
| ASR confuses numbers | TTS teen/tens fixtures and disputed-span tests |
| Spreadsheet float error | Decimal and independent-parity tests |
| Duplicate or missing IDs | Submission preflight property tests |
| Prompt injection changes behavior | Adversarial source-content tests |
| Provider interruption loses progress | Checkpoint/resume chaos tests |
| Same flawed verification path repeated | Retrieval-diversity policy tests |

## Change-control rule

No implementation story may weaken a source requirement merely to make the system simpler. If a requirement is deferred, its story remains incomplete and any dependent milestone gate remains open.
