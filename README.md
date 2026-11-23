# Northstar Atlas

**Enterprise policy Q&A with traceable evidence and explicit policy authority.**

Northstar Atlas helps employees, managers and enterprise support teams find policy
answers across HR, IT Operations, Platform Engineering, SRE, Information Security
and Compliance. Users ask a question in a Streamlit chat interface, receive a
newly generated answer, clarification or evidence-limited abstention, and inspect
the supporting policy citations.

Policy responsibilities often cross departments: business approval, Security
authorization and technical provisioning may belong to different teams. Atlas
connects those sources while preserving conditions and approval status. It helps
people find guidance; it does not grant access, approve changes or execute actions.

Interactive **Standard RAG on Mac with local Chroma, pinned MiniLM and Nebius has
been demonstrated live**, including an accepted cited HR transfer-notice answer.
The optional **Pinecone + MiniLM + Nebius** path also completed one verified live
Standard RAG question on 21 November 2026. Its reviewed 189-record publication was
ready, and the accepted answer cited HR-001 v1.0 §4.2. These observations do not
establish general answer accuracy or confirmed billing. The optional
semantic LangGraph workflow is implemented and tested offline; successful complete
live execution has not been verified. All policies are synthetic.

## How it works

![Standard RAG architecture showing six synthetic policies (one approved HR baseline and five proposed drafts), MiniLM embeddings, Chroma or Pinecone retrieval, and controlled Nebius generation and evidence validation.](docs/assets/standard-rag-architecture.png)

*Both demonstrated Mac paths use pinned local MiniLM embeddings and controlled
Nebius generation, with Chroma or an independently approved Pinecone publication
for retrieval. Readiness and authorization are checked for each installation;
the successful live tests do not provision assets or grant access to a clone.*

The linear `QAService` defaults to dense retrieval and local Chroma; Pinecone is
explicitly selectable. BM25 and dense/BM25 Reciprocal Rank Fusion are optional.
Indexing is separate from answering questions.

The UI separates three experiences:

- **Offline demo:** clearly labeled authored examples; no retrieval or inference.
- **Policy search — retrieval only:** free-text search over real local Chroma
  evidence, with citations and relevance scores; no generated answer.
- **Live Q&A:** generated answers, Sources, and expandable Evidence and Validation
  details. Questions are independent; displayed chat history is not added to prompts.

Provider failures and rejected responses are withheld as safe errors, not presented
as policy abstentions. See the [public architecture guide](docs/README.md#current-component-boundaries)
for component contracts and engineering decisions.

## Technology stack

| Layer | Implementation |
| --- | --- |
| Runtime | Python 3.12, uv, pinned dependencies in `uv.lock` |
| Interface | Streamlit |
| Embeddings | Local CPU `sentence-transformers/all-MiniLM-L6-v2`; pinned revision, normalized 384-dimensional vectors |
| Retrieval | Chroma locally; optional managed Pinecone; standard-library BM25 and hybrid RRF |
| Generation | Nebius Token Factory, `Qwen/Qwen3-30B-A3B-Instruct-2507`, through an HTTPX adapter |
| Validation | Provider-independent `grounded-answer-v2` contract and deterministic evidence checks |
| Optional orchestration | LangGraph with bounded routing, retrieval, grading and semantic-checking profiles |
| Tests | Standard-library unittest, fake providers and opt-in cached-model integrations |

## Synthetic corpus and policy governance

The [corpus manifest](data/corpus_manifest.json) verifies six source documents in
[data/policies](data/policies/). Heading-aware processing produces 151 canonical
chunks; MiniLM-aware refinement produces **189 embedding-ready records**, retaining
source spans and sibling relationships. The recorded Pinecone publication verified
that 189-record snapshot. MiniLM's 256-token input limit is enforced without silent
truncation.

| Policy | Domain | Status |
| --- | --- | --- |
| HR-001 — Employee Role Change & Offboarding | Human Resources | v1.0, Approved (Synthetic), effective 2026-10-01 |
| IT-001 — Identity & Access Management | IT Operations | v0.1, Proposed |
| PLAT-001 — Production Deployment & Change Management | Platform Engineering | v0.1, Proposed |
| SRE-001 — Incident Response & Escalation | SRE | v0.1, Proposed |
| SEC-001 — Privileged Access & MFA | Information Security | v0.1, Proposed |
| COMP-001 — Access Review & Audit Evidence | Compliance | v0.1, Proposed |

Policy status is intentional synthetic test data: HR-001 is the approved baseline;
IT-001, PLAT-001, SRE-001, SEC-001 and COMP-001 remain proposed so the application
can demonstrate approved-only versus draft-review governance. Proposed does not
mean the application is unfinished.

**`approved_only`** admits approved policies effective on the reference date.
**`draft_review`** also admits proposed drafts, preserving their labels in evidence
and responses. A draft is not an effective company rule, even when an approved
policy refers to it.

![Policy governance: approved-only excludes drafts; draft-review retains proposed labels. Evidence validation leads to cited answers, evidence-limited abstentions or withheld failures, without proving semantic correctness.](docs/assets/policy-governance.png)

Retrieved text is treated as untrusted evidence. Source hashes, citation identity,
version, status and required siblings are checked before generation. Context
assembly preserves complete evidence groups within the prompt budget. Citations
must refer to supplied eligible evidence; quotations must match contiguous source
text exactly or after whitespace normalization.

Caller-reviewed obligations can require specific evidence and statements; ordinary
questions retain coverage `not_assessed`. **Valid citations and quotations do not
prove semantic correctness or completeness.** A documented policy gap differs
from insufficient retrieval evidence.

## Quick start

Use an existing **Python 3.12** interpreter and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The commands below are for macOS/Linux shells, run from the repository root.
Keep the environment and caches inside the checkout:

```bash
pwd -P
export NORTHSTAR_PROJECT_ROOT="$PWD"
mkdir -p .local/tmp
export UV_CACHE_DIR="$PWD/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.local/uv-python"
export UV_PYTHON_BIN_DIR="$PWD/.local/bin"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export TMPDIR="$PWD/.local/tmp"
export PYTHONPYCACHEPREFIX="$PWD/.local/pycache"
uv sync --locked --python 3.12 --no-python-downloads --extra ui --extra llm
```

This installs the local retrieval stack, UI and generation support. Add
`--extra pinecone` or `--extra orchestration` as needed; use `--all-extras` for
development. Model and index provisioning are separate. The application validates
its root, manifest and source hashes.

### Offline demonstration

```bash
export NORTHSTAR_UI_ENABLE_LIVE=0
export NORTHSTAR_UI_ENABLE_PINECONE=0
.venv/bin/python -B -m streamlit run src/northstar_atlas/ui/app.py
```

Open the local URL and select **Offline demo**. Its authored examples need no
provider configuration, cached MiniLM model or vector index. The local server
binds to loopback, with CORS/XSRF protections enabled and telemetry disabled.

### Local policy search

For real retrieval, validate the corpus, explicitly download the pinned model once,
and build the local Chroma index:

```bash
.venv/bin/python -B scripts/validate_evaluations.py
.venv/bin/python -B -m northstar_atlas.ingestion --mode draft_review
.venv/bin/python -B -m northstar_atlas.local_embeddings download
.venv/bin/python -B -m northstar_atlas.vector_store index
.venv/bin/python -B -m northstar_atlas.vector_store status
```

Skip downloading and indexing when verified assets are already ready. Runtime
embeddings use cached model files. Choose **Policy search — retrieval only** in
Streamlit to search up to five passages with citations and similarity scores.
Similarity measures retrieval relevance, not answer confidence.

## Configuration and controlled-live use

[`.env.example`](.env.example) documents supported settings and nonsecret defaults.
If using a local `.env`, create it only when absent and configure it privately;
never overwrite an existing file. Alternatively, supported values can come from
the process environment. Conflicting values are rejected. Reserved settings in the
example are not runtime overrides; MiniLM and the local storage layout remain fixed.

Live generation requires separately configured Nebius access, the reviewed model,
current pricing and explicit authorization. Server flags are process-environment
settings, not `.env` settings. The UI also requires session consent. Enabling a flag
alone is not sufficient for controlled hosted use.

### Optional Pinecone

The HTTPX Documents list/fetch path verified **189 of 189 records**. A separately
approved no-upsert restoration then freshly verified record integrity and exact
query visibility for all 189 records before promoting only the local manifest.
The active pointer was preserved. Retrieval-only search and one Pinecone-backed
Nebius answer subsequently succeeded; the reviewed publication remained ready.

These are point-in-time checks of one controlled installation, not unrestricted
production-readiness evidence. Another checkout requires its own verified assets
and approvals. Startup never restores readiness. Integrity failures can mark local
publication metadata incomplete and withhold a result; they do not authorize
remote vector changes, repair or retry. Historical SDK comparisons remain separate.

Install the `pinecone` extra and supply the four required process variables when
`.env` is absent: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD` and
`PINECONE_REGION`. All must be complete and valid. When `.env` exists, its security,
containment, Git-exclusion, parsing and environment/file conflict checks remain
mandatory; environment values cannot repair an incomplete file.

Both paths require an explicitly reviewed target, owner confirmation,
credential-fingerprint binding and compatible publication metadata. The adapter
verifies source/model identity and audits the remote index. It does not silently
select an account or provision resources. Pinecone still requires local MiniLM
for query embeddings. See the [retrieval and configuration guide](docs/README.md)
for the adapter and its verification boundaries.

### Controlled batches

The historical hosted controlled interactive mode accepts user-entered questions through **Standard
RAG → Pinecone**, with either policy mode. Each separately approved batch permits
five submissions:

- Reserve the full bounded cost before provider I/O, with durable shared limits
  across sessions, processes and restarts.
- Reject duplicate invocation IDs and retain conservative reservations for uncertain
  outcomes; do not automatically retry failed requests.
- Display remaining capacity and disable submission when exhausted or unavailable.
- Settle accounting and durably record the accepted public result before display.

Private-preview verification, current pricing, operator approval and session
consent are required. Clearing chat or restarting does not renew the allowance.
The simpler local UI attempt counter is process-local and is **not** a durable
spending cap or an access-control system.

For an already provisioned and separately authorized controlled Replit preview,
run `bash scripts/run_replit_live.sh` using the [controlled-live launcher](scripts/run_replit_live.sh).
Its preflight checks must pass before Live Q&A is available. The default [.replit](.replit) launcher remains
offline-only. See [hosted and legacy compatibility](docs/README.md#hosted-and-legacy-compatibility)
for the activation boundaries; installation alone does not authorize
paid use.

### Controlled local Mac runtime

For a trusted single-user Mac, the repeatable development profile supports Chroma
and, after a separately approved capability amendment, Pinecone. Both use the same
independently approved durable development authority. It does
not use Replit Private Preview settings or fall back to a historical authority:

```bash
bash scripts/run_local_live.sh --approve-local-live --development --backend chroma

# Alternative: only after the Pinecone capability is approved and applied.
bash scripts/run_local_live.sh --approve-local-live --development --backend pinecone
```

Run only one backend launcher at a time, after separate paid-use authorization and configuration of an
**existing** compatible authority, current independently verified pricing, cached
MiniLM and verified retrieval assets. Select the latest applied plan with
`NORTHSTAR_DEVELOPMENT_PLAN` and derive `NORTHSTAR_DEVELOPMENT_RUN_APPROVAL`
through `local_development.run_approval` as described in the
[development procedure](docs/decisions/Local-Development-Authority.md#preparation-activation-and-execution).
An old plan or a digest without operator approval does not authorize execution.
Open `http://127.0.0.1:8501` and give session
consent. Missing accounting or expired pricing stops startup; the launcher never
creates a ledger or restores remote history. The development profile permits at
most 100 lifetime attempts and $2.00 cumulative reserved estimated cost, with
pricing reviewed for one calendar month. These are configured ceilings, not a
statement of current balance or account-wide spending. Renewal never resets usage.
The lifetime attempts and Nebius reservation ceiling are shared across both
backends; switching backends creates no new allowance. Pinecone read and egress
charges are outside that ceiling. The launcher fixes the backend for the session;
both `approved_only` and `draft_review` remain available. Retrieval-only Policy
Search remains Chroma-only. This profile disables LangGraph. Other separately
authorized profiles retain their own contracts. Do not tunnel or publicly forward this local service.
See the [development setup and approval procedure](docs/decisions/Local-Development-Authority.md)
and [local access contract](docs/decisions/Local-Controlled-Live-Runtime.md).

## Recorded evaluation

The [frozen 15-question dataset](evaluations/README.md) covers precise HR lookups,
cross-policy scenarios, ambiguous questions and unsupported-value requests. The
24 October 2026 retrieval evaluation used the same source snapshot and an explicit
eligibility reference date.

![Expanded evidence recall across 15 questions: Dense 64.67%, BM25 66.78%, Hybrid RRF 70.44%. Every strategy fully covers six questions; retrieval coverage is not answer accuracy.](docs/assets/retrieval-recall-comparison.png)

| Strategy | Direct Recall@5 | Recall after sibling expansion | Full expanded evidence coverage |
| --- | ---: | ---: | ---: |
| Dense | 55.22% | 64.67% | 6/15 |
| BM25 | 51.56% | 66.78% | 6/15 |
| Hybrid RRF | 57.44% | 70.44% | 6/15 |

Hybrid recovered more required evidence on average, but did not improve complete
coverage and caused more full-prompt budget failures in this experiment. Dense
remains the default. Chroma and Pinecone returned identical top-five ID sets in
the recorded backend comparison, with two ordering differences.

These are **retrieval coverage scores, not answer accuracy or faithfulness**.
See the [retrieval and bounded-graph evaluation](docs/decisions/Retrieval-and-Bounded-Graph-Evaluation.md)
and its [historical backend comparison](docs/decisions/Retrieval-and-Bounded-Graph-Evaluation.md#appendix-b--historical-chroma-pinecone-comparison) for
methodology, timing, missing evidence and failures.

The Mac/Chroma live observations include an accepted HR notice answer and an AKS
request that timed out after one generator transport attempt. No answer was
accepted for the timeout; its uncertain settlement retained the full reservation,
and provider completion/billing remained unknown. Another HR answer passed
citation/quotation validation but incorrectly assigned role-change approval to HR;
the policy assigns approval to the receiving manager and recording to HR.

A targeted, offline-tested [HR actor safeguard](docs/decisions/Reviewed-Actor-Validation.md)
now restricts claims citing the exact reviewed HR-001 v1.0 §4.1 snapshot to supplied
source spans or reviewed wording. It withholds unsupported or unreviewed wording
without rewriting answers or adding model calls. It may reject correct paraphrases
and does not prove general semantic correctness or completeness. A successful live
retest of this safeguard is not established.

The verified Pinecone + Nebius test asked how far in advance HR must notify IT
Operations of a planned internal transfer. One Qwen generator attempt returned
the accepted **at least two business days** answer with HR-001 v1.0 §4.2 and an
exact supporting quotation. Sanitized evidence recorded 4,005 input and 160 output
tokens, durable settlement and result persistence. This single approved-only
example is not an answer-quality benchmark, a test of the §4.1 actor safeguard,
or confirmation of provider billing totals.

Historical Replit/Pinecone observations include accepted cited answers, an
evidence-limited abstention and a withheld HTTP 200 response-envelope failure.
That envelope defect remains unresolved; it is not identified as the later Mac
timeout. None of these observations is a live fifteen-question answer benchmark.
Offline examples and fake-provider graph tests are separate evidence.

## Optional LangGraph workflow

LangGraph is opt-in; Standard RAG remains the default. The profiles have different
behavior and call bounds:

| Profile | Behavior | Maximum LLM calls |
| --- | --- | ---: |
| `bounded_v1` | Deterministic routing and evidence assessment around retrieval and generation | 1 |
| `llm_routing_grading_v1` | LLM category routing and per-chunk relevance grading before generation | 4 |
| `llm_routing_grading_semantic_v1` | Adds semantic checking and one possible bounded regeneration | 7 |

The semantic workflow includes routing, retrieval, relevance grading, answer
generation and semantic checking. It supports Chroma and Pinecone, at most two
retrieval attempts and at most one regeneration after an unambiguous negative
semantic check. Deterministic validation always takes precedence; malformed,
unclear or exhausted checks withhold the candidate.

Conditional transitions and failure handling have been tested offline. Successful
complete live semantic-LangGraph execution has **not** been verified. Live use
requires separate authorization and reservation of the full workflow before
provider I/O. Future work is live end-to-end validation and comparison with
Standard RAG for answer quality, latency and cost. See the
[graph profiles and backend contracts](docs/README.md#optional-langgraph-profiles).

## Tests

With all development extras installed:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python -B scripts/test_offline.py
.venv/bin/python -B scripts/validate_evaluations.py
```

Tests use fake providers, not paid inference. Cached-model checks require the
pinned snapshot and `NORTHSTAR_RUN_MODEL_TESTS=1`; historical replays may skip when
local artifacts are absent.

Graph asset and recovery tests construct deterministic snapshots from the public
synthetic policies; no private snapshot is required. Their authored vectors test
integrity and control flow, not MiniLM output or retrieval quality.

The [offline integrity job](docs/decisions/Offline-Integrity-Checks.md) also checks
source syntax, direct dependency pins and frozen evaluation integrity using only
Python 3.12's standard library. It supplements the full application suite; it does
not verify browser interaction or live providers.

## Limitations and next steps

- Six synthetic policies and fifteen development questions are a small, repeatedly
  inspected dataset. No held-out answer-quality or faithfulness score is established.
- Cross-policy evidence can be incomplete, including temporary AKS access ownership.
  Sibling expansion cannot recover source sections that were never retrieved.
- Valid source references can accompany unsupported free-form claims. Human review
  remains necessary for material policy decisions; semantic checking also depends
  on the evidence actually retrieved.
- Large evidence groups can exceed the complete prompt budget and block generation.
  The recorded provider-envelope failure and live semantic-graph validation remain open.
- Full-stack Linux installation, memory, startup and persistence need environment-
  specific verification. The locked Torch stack can pull substantial Linux wheels.
  The separate 35-package Replit UI-only setup supports authored previews, not live RAG.

Planned work includes independently reviewed held-out questions, human answer
assessment, better cross-policy candidate coverage, measured reranking experiments
and live graph comparison. Controlled demonstration does not establish readiness
for unrestricted production policy advice.

[Architecture and experiments](docs/README.md) · [Source](src/northstar_atlas/) ·
[Tests](tests/) · [Evaluation assets](evaluations/)

## Copyright and Permissions

Copyright © 2026 Anvesh Rao. All rights reserved.

This repository is made available for portfolio and evaluation purposes. No general license is granted to use, modify, redistribute, sublicense or commercially exploit the original project source code, except as permitted by applicable law or platform terms. Third-party components remain subject to their respective licenses.
