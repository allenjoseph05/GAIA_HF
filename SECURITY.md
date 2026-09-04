# Security model

GAIA Max is an evaluation and portfolio system, not a multi-tenant execution
service. Its safety goal is to fail closed when inputs, evidence, tools, output
contracts, or submission authority are uncertain.

## Trust boundaries

- Application configuration, task profiles, solver registrations, output contracts,
  and tool policies are trusted code/operator inputs.
- Questions, attachments, web pages, model responses, captions, OCR, and tool outputs
  are untrusted data.
- Candidate answers and evidence are private runtime data.
- The public Space is a read-only demonstration and has no submission capability.

## Enforced invariants

- Untrusted source text cannot add a tool to a code-owned allowlist.
- Tool calls have call-count, input, output, timeout, and concurrency budgets.
- Model-oriented source context is scanned, quarantined, size-bounded, and represented
  as data in a user message separate from the system policy.
- Unsupported or ambiguous routes, missing evidence, invalid output schemas, stale
  snapshots, and unresolved verification conflicts block progress.
- Submission defaults off and requires an approval bound to an exact frozen manifest
  hash. Submission is kept outside the public application.
- Logs and public reports contain status, hashes, counts, and reason codes rather than
  official candidate answers, source content, or credentials.

## Known limitations

- Prompt-injection scanning is defense in depth, not a proof that arbitrary text is
  benign. The primary boundary is structural separation plus code-owned capabilities.
- The Python specialist uses AST restrictions and disposable bounded subprocesses. It
  is not a VM-grade sandbox and must not execute adversarial code in a multi-tenant
  service.
- URL validation rejects literal non-global addresses, localhost names, credentials,
  unsafe schemes, and unsafe redirects. A production multi-tenant deployment should
  additionally pin DNS resolutions and validate the connected peer to resist DNS
  rebinding.
- The public end-to-end demo covers the completed deterministic solver stack. Other
  research/media modules are reusable components, not a claim of universal autonomous
  solving.
- No software can enumerate every edge case. This project instead documents supported
  inputs, enforces invariants, uses property/adversarial tests, and fails closed outside
  its supported envelope.

Report suspected vulnerabilities privately to the repository owner. Do not include
tokens, private candidates, gated dataset content, or benchmark answers in an issue.
