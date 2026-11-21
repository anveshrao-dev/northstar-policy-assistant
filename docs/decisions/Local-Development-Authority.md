# Repeatable local Standard RAG development

This opt-in Mac workflow supports independent Standard RAG questions using Chroma
or an explicitly approved Pinecone capability, pinned local MiniLM embeddings and
the Nebius Qwen generator. The Mac/Chroma path has been demonstrated live; one
Pinecone-backed approved-only HR notice answer was also verified live on
21 November 2026. Synthetic regression tests independently
verify its controls; neither observation establishes general answer correctness.
The default Offline Demo, Policy Search, hosted Replit, one-shot and LangGraph
contracts remain unchanged. LangGraph is not enabled by this mode. The initial
development authority is Chroma-only; Pinecone requires the separate amendment below.

## Accounting and access

The fixed private development root is `.local/development-authority/`; its
accounting files live in its own `.local/controlled-live/` subdirectory. This
reuses the existing authority-root, ownership, permission, identity, exclusive
lock, atomic write and settlement mechanisms without opening the original
project's `.local/controlled-live/` authority. There is no fallback or migration.
Preparation generates a distinct scope and billing identity; existing directories
are refused, including incomplete preparations. A missing authority is an error,
not permission to initialize one during startup.

The development policy allows at most **100 durably reserved question attempts**
and **$2.00 cumulative reserved estimated cost**, whichever is exhausted first.
Every reservation retains the full 30,720-input/2,048-output generator envelope,
including retrieval/setup failures, abstentions, rejected answers and uncertain
provider outcomes. Counts and estimates are never refunded by this profile.
The estimate is a conservative local accounting bound, not a billing statement
or an account-wide Nebius spending cap. Other projects and hosts can spend
independently.

After a Pinecone capability is applied, both backends share these same cumulative
limits. No new attempts are created and no historical uncertain reservation is
released. Pinecone read and egress charges are outside the $2 Nebius reservation
ceiling; neither ceiling establishes confirmed billing totals.

Each question receives a unique invocation ID and one single-use transport
permit. Duplicate IDs fail across sessions and restarts; no automatic paid
retry occurs. The accounting lock spans reservation, resource setup, generation
and settlement. A contending process is denied rather than queued. Invalid
responses are withheld. Settlement must finish and the existing private public
result recorder must persist the result before the UI displays it. Failed
recording cannot trigger another request with the same ID.

The launcher requires a trusted single-user Mac and binds only to `127.0.0.1`.
The existing direct-loopback client, exact Host/Origin, no-forwarding, XSRF,
operator-approval and browser-session consent checks remain mandatory. Hosted
Private Preview is not simulated or bypassed. The development selection cannot
be combined with one-shot, handoff, batch, external-authority or generic budget
settings. Selection changes during a request are rechecked before transport.

## Calendar-month pricing

Only this coordinator uses monthly pricing. The existing global 24-hour pricing
validator and historical budget ceilings are unchanged.

An independently reviewed price has six fields: `id`, `input_per_million_usd`,
`output_per_million_usd`, `source`, `confirmed_at`, and `valid_until`. Rates are
positive decimal strings. The source must be the model-specific official page:
[Qwen model pricing](https://nebius.com/services/token-factory/models/qwen-models-inference).
The operator must independently verify the exact configured
`Qwen/Qwen3-30B-A3B-Instruct-2507` model and rates; a valid URL or digest alone
is not evidence of that verification.

Timestamps are timezone-aware. Expiry is calculated from the actual verification
instant converted to UTC: the same UTC clock time in the following calendar
month, with the day clamped to that month's last day. January 31 becomes February
28 or 29; March 31 becomes April 30. This is not a 30-day duration. At expiry,
paid requests stop. No lookup, provider response or successful request silently
refreshes a price or extends an approval.

A separately approved renewal appends one unique pricing revision, retaining all
old revisions and workflow history. It recalculates affordability for the next
full request under the new rates. It never increases limits or resets counts.
The candidate is bound to exact prior ledger bytes, ledger device/inode,
authority/lock identities, corpus, embedding revision, model and previous plan.
Unsettled reserved workflows block renewal; completed uncertain workflows keep
their full consumption. A prepared but unapplied renewal fences the previous
runtime approval until independent application review.

## Preparation, activation and execution

These are command templates, not permission to create or activate accounting.
First separately approve the distinct scope and account-wide exposure, stop
competing development writers/restarts, and review current official pricing.
Keep pricing/review artifacts under ignored `.local/`, files 0600 and directories
0700. The existing project `.local/` directory must already be safe. Do not use
real authority files as test fixtures.

```bash
.venv/bin/python -B -m northstar_atlas.local_development prepare \
  --pricing-file "$PRIVATE_PRICING_FILE" \
  --verified-pricing-sha256 "$REVIEWED_CANONICAL_PRICING_SHA256" \
  --scope-review-sha256 "$REVIEWED_SCOPE_EVIDENCE_SHA256" \
  --writers-stopped --approve-preparation
```

Inspect the inert `review-0000` plan, candidate and source snapshot. Independently
review the identity, source, model and limits. Approve the exact returned plan
and candidate digests before application:

```bash
.venv/bin/python -B -m northstar_atlas.local_development apply \
  --approved-plan-sha256 "$APPROVED_PLAN_SHA256" \
  --approved-candidate-sha256 "$APPROVED_CANDIDATE_SHA256" \
  --writers-stopped --approve-application
```

Verify the applied receipt, immutable initial activation, matching ledger,
current pricing and absence of pending guards. This activates accounting but
neither launches Streamlit nor grants browser consent or provider access.
Separately approve local paid execution against the **latest applied plan**.
Select its digest explicitly and derive the corresponding run approval through
the existing contract. The shell variable below must contain the independently
reviewed digest; it is not generated by startup:

```bash
export NORTHSTAR_DEVELOPMENT_PLAN="$LATEST_APPLIED_PLAN_SHA256"
export NORTHSTAR_DEVELOPMENT_RUN_APPROVAL="$(
  .venv/bin/python -B -c 'import os; from northstar_atlas.local_development import run_approval; print(run_approval(os.environ["NORTHSTAR_DEVELOPMENT_PLAN"]))'
)"
```

Possessing or calculating a hash is not operator approval. After an amendment or
pricing renewal, both variables must select that new applied plan; missing,
stale or mismatched values deny startup. Existing credential configuration is
loaded only through the guarded provider path, after reservation.

```bash
bash scripts/run_local_live.sh --approve-local-live --development --backend chroma

# Use instead only after separately applying the Pinecone capability.
bash scripts/run_local_live.sh --approve-local-live --development --backend pinecone
```

Open `http://127.0.0.1:8501`, select Live Q&A, choose approved-only or draft-review,
and explicitly enable paid answers for the session. Each submission has its own
question and evidence; display history is not added to model prompts. The sidebar
shows shared remaining question and cost capacity. A disabled or unavailable
budget cannot be replaced with a process-local allowance.

Run only one launcher at a time. Its backend selection is fixed for that session;
it is not a browser-controlled authority override. Retrieval-only Policy Search
continues to use Chroma and does not generate answers. A first integration demo
can use approved-only mode and the question “For a planned internal transfer,
how far in advance must HR notify IT Operations?” Inspect the generated answer,
HR-001 v1.0 §4.2 quotation and validation details, then disable paid consent.
That procedure requires separate live approval; a failed or uncertain response
does not authorize another submission.

## Optional Pinecone capability amendment

The amendment extends the existing development authority without replacing it.
Original activation and approvals remain immutable. Its candidate ledger is
byte-identical to the source: scope, billing identity, pricing, workflow history,
reservations and ceilings do not change. Chroma remains permitted, and both
`approved_only` and `draft_review` are supported. Only Standard RAG is enabled.

The capability binds the ready, owned 189-record publication: target,
owner-confirmation metadata, namespace, frozen snapshot, corpus and relevant file
hashes and filesystem identities. It also binds the digest of a separately reviewed
external scope document. The application does not define or enforce that
document's schema; its contents require independent operator review. The review
must acknowledge Pinecone charges, unchanged limits and writer requirements.

With competing development writers stopped, separately approve preparation using
the current applied parent plan and the exact reviewed scope-document digest:

```bash
.venv/bin/python -B -m northstar_atlas.local_development prepare-pinecone \
  --publication-id "$REVIEWED_PUBLICATION_ID" \
  --previous-plan-sha256 "$CURRENT_APPROVED_PLAN_SHA256" \
  --scope-review-sha256 "$REVIEWED_CAPABILITY_SCOPE_SHA256" \
  --writers-stopped --approve-preparation
```

Preparation creates the next private immutable `review-NNNN` packet, without
credentials, provider operations or ledger changes. Even a partial packet blocks
the previous runtime approval. Review the source, candidate and plan before
separately approving application with their exact digests:

```bash
.venv/bin/python -B -m northstar_atlas.local_development apply \
  --approved-plan-sha256 "$APPROVED_AMENDMENT_PLAN_SHA256" \
  --approved-candidate-sha256 "$APPROVED_AMENDMENT_CANDIDATE_SHA256" \
  --writers-stopped --approve-application
```

Current pricing, authority continuity, unchanged accounting and publication
bindings must pass validation. An unresolved reserved workflow or pending guard
blocks the operation. The original exclusive accounting lock and durable
application safeguards remain in force. Application neither starts the server
nor grants live consent. Verify its receipt and unchanged cumulative accounting,
then separately authorize execution and select the new plan/run approval.

Pinecone setup, search and generator boundaries recheck the approved binding.
Changed metadata, snapshot, file identity, target or pending restoration blocks
execution; there is no fallback to another publication. Failure after reservation
retains conservative accounting and cannot bypass settlement. A stale Pinecone
publication does not prevent the separately selected Chroma path or a valid
pricing renewal. Subsequent pricing revisions preserve the capability and history.
See the [restoration contract](../README.md#guarded-readiness-restoration) for
separately approved publication operations; no restoration happens during Q&A.

## Renewal and interrupted operations

With writers stopped, independently verify new pricing and separately authorize
preparation. Use a unique price ID and a later verification timestamp:

```bash
.venv/bin/python -B -m northstar_atlas.local_development prepare-renewal \
  --pricing-file "$PRIVATE_NEW_PRICING_FILE" \
  --verified-pricing-sha256 "$REVIEWED_NEW_PRICING_SHA256" \
  --previous-plan-sha256 "$CURRENT_APPROVED_PLAN_SHA256" \
  --writers-stopped --approve-preparation
```

Review the new immutable packet and approve its exact digests. Apply using the
same `apply` command with those new digests, then separately approve the new
execution selection. Old plan/run approvals are rejected. Reapplying a receipt
or reusing a price ID is denied.

A pending settlement, partial packet, identity change, corrupt ledger or failed
write blocks execution. Keep writers stopped after any interruption. Do not
retry, delete guards, restore an older ledger, recreate the root or automatically
roll back. Recovery needs independent state/storage review and separate approval.
Filesystem failure during guard restoration cannot be remedied by application
logic alone. Locks coordinate cooperating local writers, not hostile same-user
code, old binaries, copied authorities or independent hosts.

## Validation scope

Synthetic tests exercise calendar boundaries, review/apply/renew approvals,
immutable spending, 100-question and cost exhaustion, duplicate/restart denial,
lock contention, failure guards, local consent and Streamlit presentation.
Pinecone amendment tests additionally cover stale approval, changed publication
bindings, shared accounting, no provider access before reservation and consent,
and Chroma continuity. Restoration tests use synthetic HTTP responses and check
all-record visibility, request caps and interrupted transitions. Cached
integration checks use real isolated Chroma/MiniLM retrieval and fake Nebius HTTP
for three sequential answers, draft evidence, abstention and withheld failures.
No real authority, current pricing approval, credential or paid response is
established by these tests. Deterministic citation and quotation validation does
not prove every model statement semantically correct.

See [local access](Local-Controlled-Live-Runtime.md),
[Chroma isolation](Secure-Chroma-Initialization.md), and
[hosted and legacy compatibility](../README.md#hosted-and-legacy-compatibility).
