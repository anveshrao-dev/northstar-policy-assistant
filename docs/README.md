# Architecture and verification guide

Northstar Atlas is an enterprise policy Q&A application over six synthetic policy
documents. Its demonstrated live path is Streamlit → local Chroma and pinned
MiniLM → Nebius Qwen → validated public response. The [README](../README.md)
provides installation and everyday usage; this guide explains component boundaries
and the evidence supporting current claims.

## Current component boundaries

| Component | Responsibility |
| --- | --- |
| Source corpus | Six allowlisted Markdown policies, version/status metadata and SHA-256 hashes in the corpus manifest. Evaluation answers and documentation are never indexed. |
| Ingestion | Heading-aware loading and canonical chunking preserve source spans, tables, conditions and document/section identity. |
| MiniLM refinement | 151 canonical chunks become 189 embedding records. Oversized families are refined to the pinned tokenizer's 256-token limit, including special tokens; inputs are not silently truncated. |
| Embeddings | Local CPU `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; normalized 384-dimensional vectors. |
| Retrieval | Local Chroma is the default. Optional Pinecone publications bind source/model/snapshot identity. BM25 and dense/BM25 RRF are explicit alternatives. |
| Evidence assembly | Eligibility, provenance and required siblings are checked; source spans are deduplicated and whole evidence groups must fit the prompt budget. |
| Generation | Nebius Token Factory `Qwen/Qwen3-30B-A3B-Instruct-2507` through HTTPX, with a durable single-use stage permit and one transport attempt. |
| Validation | Provider envelope, structured answer shape, citation identity, quotation matching, policy status and any explicitly reviewed obligations are checked before publication. |
| Accounting and results | Reserve before resource/provider I/O; settle conservatively; persist the typed public result before display. Failure at a required persistence boundary withholds the answer. |
| UI | Authored Offline Demo, retrieval-only Policy Search, and separately authorized Live Q&A. Displayed history is not added to generation prompts. |

Index preparation is separate from answering. Startup does not download models,
rebuild indexes, repair publications, initialize an authority or renew pricing.
The installed source root and corpus are validated; a process-level
`NORTHSTAR_PROJECT_ROOT` cannot be supplied by a browser request or dotenv file.

## Policy authority and evidence

HR-001 v1.0 is approved synthetic evidence, effective 2026-10-01. IT-001, PLAT-001,
SRE-001, SEC-001 and COMP-001 are proposed synthetic drafts. They represent their
respective departments' responsibilities, not effective rules of a real company.

`approved_only` permits effective approved sources on the selected reference date.
`draft_review` also permits proposed sources and retains their labels. A reference
from an approved document to a draft does not approve the draft. The
[corpus manifest](../data/corpus_manifest.json) and [policy files](../data/policies/)
are the source of truth for status and exact bytes.

Citation checks establish that references identify eligible supplied evidence.
Quotations must match contiguous source text exactly or after whitespace
normalization. These checks do not establish that every free-form claim follows
from its quotation. Ordinary questions retain completeness `not_assessed`.
Caller-reviewed obligations can impose narrower evidence and statement contracts;
the [HR actor safeguard](decisions/Reviewed-Actor-Validation.md) rejects demonstrated
approval-role substitutions for the pinned HR-001 §4.1 obligation. It is not a
universal actor/action or semantic-entailment verifier.

An evidence-limited abstention is different from a timeout, invalid response or
failed resource setup. Those failures publish safe errors and no accepted answer.

## Retrieval and configuration

Chroma stores local supplied vectors and source metadata. Its isolated worker
startup prevents Chroma's import-time settings initialization from reading an
unrelated project dotenv file. See [secure initialization](decisions/Secure-Chroma-Initialization.md).

Pinecone is optional. A bounded remote diagnostic verified all 189 records through
the HTTPX Documents list/fetch path. The current HTTPX search implementation is
tested offline but remains live unverified; the publication remains incomplete.
A successful record audit does not establish query visibility or restore readiness.
Historical SDK search comparisons are separate experiments, not production-readiness
evidence for the current adapter. No startup path silently repairs a publication.

Pinecone requires complete `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
`PINECONE_CLOUD` and `PINECONE_REGION` settings. Process-only configuration is
supported when `.env` is absent. If the file exists, containment, Git-exclusion,
duplicate/format and environment-conflict checks remain mandatory; process values
cannot repair an invalid file. Private target metadata and credential-bound owner
confirmation must match the selected index before access. The reviewed dense-vector
profile uses 384-dimensional cosine vectors without integrated embeddings.

The HTTPX Documents adapter rejects unexpected response fields, duplicate keys,
invalid usage and mismatched records. `read_units` is a nonnegative int32;
optional `egress_bytes` is a nonnegative int64. Booleans and noninteger values are
invalid. Unavailable egress remains unavailable, with coverage reported separately.
Dense-vector search rejects integrated-embedding usage. Validated usage is not a
confirmed monetary bill. The bounded diagnostic shares a 12-dispatch cap across
HTTPX describe/list/fetch, with no retry, redirect or SDK fallback. Ordinary
SDK control-plane workflows are distinct from that diagnostic contract.

## Controlled local execution

The [development profile](decisions/Local-Development-Authority.md) supports a
distinct authority limited to 100 lifetime question attempts and $2 cumulative
reserved estimated cost. Its price review expires after one calendar month using
UTC calendar arithmetic. Renewals retain all previous revisions, usage and limits.
Historical one-shot and hosted contracts retain their own shorter pricing policies.

The [local access contract](decisions/Local-Controlled-Live-Runtime.md) requires a
trusted single-user Mac, numeric loopback binding, direct client/Host/Origin checks,
explicit operator authorization and per-session consent. Development mode is
Standard RAG with Chroma; it does not enable Pinecone or LangGraph. Conflicting
authority/profile selectors fail closed. Restarting or clearing the chat cannot
reset consumption. Independent clients or hosts can spend outside this local cap.

## Optional LangGraph profiles

| Profile | Implemented bound and behavior |
| --- | --- |
| `bounded_v1` | At most two configured retrieval attempts and one generator call, with deterministic assessment/validation and optional reviewed requirements. Nonempty evidence alone does not prove completeness. |
| `llm_routing_grading_v1` | Structured routing and relevance grading; at most two retrieval attempts and four total model calls. Available for offline/programmatic validation; not the repeatable development profile. |
| `llm_routing_grading_semantic_v1` | Adds evidence-bound semantic checks and one possible regeneration after an unambiguous negative check; at most two retrieval attempts and seven calls. |

Malformed, ambiguous, negative or exhausted checks cannot turn unsupported text
into a grounded result. Deterministic source, citation and authority validation
still applies. Chroma and Pinecone semantic retrieval adapters exist, but successful
complete live semantic-graph execution has not been verified. Controlled semantic
execution requires its own authority and the full seven-stage reservation before
provider/resource I/O; it cannot borrow the development profile's permission.

Future live comparisons should measure answer quality, latency and cost against
Standard RAG. Offline fake-provider success establishes transition and failure
contracts, not provider quality.

## Hosted and legacy compatibility

The [offline Replit launcher](../scripts/run_replit_offline.sh) supplies authored
previews. [Setup](../scripts/setup_replit_offline.sh) uses the locked 35-package UI
subset; this subset cannot run real retrieval or generation. The
[hosted live launcher](../scripts/run_replit_live.sh) retains independent Replit
Private Preview checks, source integrity and durable authorization. Local loopback
authorization never substitutes for hosted private access.

Legacy five-submission batches, one-shot requests, pricing successors and
authority-continuity modules remain implemented for compatibility. They require
their exact matching approvals, identity bindings, pricing and supported accounting
state; no new allowance or migration is implied by their presence. The legacy
scope constant is retained solely as an immutable compatibility
identifier. It is not a current account balance, credential or general billing scope.

Filesystem locks coordinate cooperating writers on one authority filesystem.
Copying an authority to another host would create independent enforcement, not a
shared account-wide limit. Historical approvals and uncertain reservations must
not be reset, copied into a fresh allowance or automatically repaired.

## Verification and limitations

- Live Mac Standard RAG produced a cited HR transfer-notice answer through local
  Chroma/MiniLM and Nebius. This is not a fifteen-question live accuracy benchmark.
- A live provider timeout produced no accepted answer and retained conservative
  uncertain accounting. New timeout subtypes are covered offline; the historical
  timeout's subtype and provider-side completion/billing remain unknown.
- A live answer incorrectly attributed role-change approval to HR despite valid
  quotations. The narrow reviewed actor safeguard was added and tested offline;
  general semantic correctness remains unproven.
- A historical HTTP 200 response-envelope failure has no established exact cause.
  Its category alone does not establish output truncation.
- The [evaluation report](decisions/Retrieval-and-Bounded-Graph-Evaluation.md)
  preserves historical retrieval scores, fake-provider graph results and failures.
  Retrieval recall and similarity are not answer accuracy or faithfulness.
- The [offline integrity job](decisions/Offline-Integrity-Checks.md) is a small
  standard-library CI subset. Full guarded tests, cached-model checks, browser
  verification and live-provider validation are distinct evidence categories.

No unrestricted production-readiness, held-out faithfulness score or live semantic
LangGraph success is claimed. Linux full-stack packaging, memory and operational
persistence require environment-specific verification.
