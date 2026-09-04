# Portfolio defense guide

## Honest project statement

GAIA Max is a typed, safety-gated workflow for solving and submitting exact-answer
research tasks. It combines deterministic specialist components with explicit evidence,
verification, serialization, checkpoint, and human-approval boundaries. The current
course evaluation scored 20/20. A public synthetic scenario proves the orchestration
stack end to end without disclosing benchmark answers.

It is not presented as a universal autonomous research agent. Open-ended research,
media, document, and chess modules have different maturity levels; only registered
runtime capabilities are executable, and missing capabilities fail closed.

## Why the system is shaped this way

| Decision | Reason | Trade-off |
|---|---|---|
| LangGraph owns control flow | Node transitions, terminal states, retries, and checkpoints are inspectable | More structure than a single agent loop |
| smolagents owns only bounded fallback reasoning | `ToolCallingAgent` supplies planning/ReAct depth without controlling submission or global routing | Open research blocks when evidence remains weak |
| LlamaIndex is a passage finder, not an authority | BM25 and optional dense fusion improve long-document discovery while exact excerpts retain provenance | Retrieval scores cannot approve a claim |
| Pydantic models every boundary | Invalid model, tool, or network data is rejected before it becomes trusted state | Extra schema maintenance |
| Deterministic tools outrank model guesses | Exact-match benchmarks punish plausible but incorrectly formatted answers | Requires specialist engineering per task shape |
| Profiles are hash-bound to questions | A profile cannot silently apply to changed benchmark wording | Profiles must be regenerated when a snapshot changes |
| Semantic answers precede strings | Solvers cannot accidentally own commas, rounding, names, or units | Serializer and contract parser become critical components |
| Verification replays an authority | A solver result is independently recomputed before approval | Approximately doubles deterministic work |
| Routes and tools use code-owned allowlists | Source or model text cannot grant itself capabilities | Unsupported requests block instead of improvising |
| Source data is separate from system policy | Prompt injection remains data rather than control input | Some suspicious source lines may be quarantined |
| Operations are resource-bounded | Calls, bytes, time, redirects, retries, and concurrency cannot grow without limit | Large legitimate inputs may need an operator-approved policy change |
| SQLite checkpoints are per task | Interrupted tasks resume while completed tasks are reused | Persistent lifecycle code is more complex |
| Submission is a separate capability | Public code and normal runs cannot accidentally submit | Requires an explicit private operator step |
| Approval binds the candidate SHA-256 | Approval cannot be reused after any answer changes | Every revision needs a new explicit approval |
| Official answers remain private | Prevents benchmark leakage and keeps the Space safe | The public demo uses synthetic data |
| OTel metadata is denylisted by construction | Graph/agent/tool/retriever latency is observable without prompts, answers, tokens, or credentials | Content-rich debugging stays local and explicit |

## End-to-end execution path

```text
Question inventory
  -> canonical snapshot and SHA-256
  -> hash-bound answer-free profile
  -> deterministic analysis reconciliation
  -> code-owned route decision
  -> registered specialist solver
  -> bounded smolagents fallback when no stronger specialist applies
  -> guarded LlamaIndex passage discovery and observed-citation catalog
  -> independent deterministic replay / evidence ladder
  -> typed semantic answer
  -> exact contract serialization
  -> final structural validation
  -> durable task checkpoint
  -> bounded concurrent run collection
  -> private 20-task preflight
  -> frozen candidate hash
  -> explicit human approval
  -> one-shot submission
```

The public `gaia run` command executes through run collection using two synthetic
deterministic questions. `gaia resume` demonstrates terminal checkpoint reuse; the
integration suite also tests recovery from an interrupted task.

## Context engineering

Context is treated as a data-engineering problem rather than one large prompt:

1. The objective comes from trusted task/profile state.
2. Retrieved bytes are size-limited and stored by content hash.
3. Extracted text is scanned and assigned an allow, sanitize, or block disposition.
4. The model-facing source artifact is distinct from the original artifact.
5. Source text appears inside a JSON `untrusted_source` field in a user message.
6. System policy, objective, tools, URL, artifact hash, injection flags, original and
   included character counts, and truncation are explicit fields.
7. Model proposals cannot contain answer-bearing keys or choose their own route.
8. Stronger profile and deterministic rules win disagreements; uncertainty requires
   review or blocks execution.

No paid model provider is forced into the default install. The smolagents adapter accepts
an injected model, validates a strict `ResearchDraft`, and runs behind the same code-owned
tool gateway. This preserves zero-cost and provider choice while shipping a real framework
integration.

## Tool enforcement

`RouteToolPolicy` binds a route to the smallest allowed tool set. `EnforcedToolGateway`
rejects unknown or unregistered tools and enforces:

- maximum calls;
- cumulative input bytes;
- output bytes per call;
- wall-clock timeout;
- maximum concurrency;
- JSON-compatible output schemas; and
- answer/content-free authorization audit events.

Adversarial tests show that source text requesting `shell` cannot extend a policy that
allows only `search`. Timeout, oversized input/output, unavailable route, malformed
attachment, prompt injection, invalid serialization, checkpoint interruption, and stale
snapshot behaviors fail closed.

## What the 20/20 score proves

The official receipt proves that the frozen set of 20 submitted answers was correct. It
also exercises exact serialization, candidate completeness checks, hash-bound approval,
and submission transport.

It does not by itself prove that every answer was generated autonomously by the public
runtime. The public demo and automated tests separately prove the integrated graph path.
This distinction is deliberate and should be stated in an interview.

## Quality evidence

- Unit, property, mutation-style, and integration tests.
- Offline end-to-end hybrid test covering LangGraph, smolagents, LlamaIndex, claim-level
  evidence approval, serialization, tool audit, and OpenTelemetry spans.
- Ruff lint gate.
- Strict Pyright gate over production code. Diagnostics caused solely by incomplete
  third-party annotations are disabled explicitly; argument and value compatibility
  checks remain enabled.
- Frozen `uv.lock` dependency graph.
- CI on Python 3.11 and 3.13 plus a Docker build.
- Docker runs as a non-root user and excludes private runtime paths and secrets.
- Public sanitized result with candidate, snapshot, and Space commit hashes.

## Limitations worth saying aloud

- Only completed and registered solver routes are runnable end to end.
- Dense document retrieval requires an injected embedding model; BM25 is always available
  in the research extra, and local Hugging Face embeddings remain a separate heavy extra.
- Prompt-injection scanning is defense in depth; code-owned authorization is the real
  capability boundary.
- Restricted Python subprocess execution is not a multi-tenant security sandbox.
- Literal private-network URLs are blocked, but multi-tenant SSRF defense should also
  pin DNS resolution and validate the connected peer.
- The project favors explicit blocking over low-confidence completion, so coverage is
  narrower than an unconstrained chatbot.

Those are controlled scope decisions, not hidden defects. A strong defense explains why
each boundary exists, how the code enforces it, and what would change for a different
deployment threat model.
