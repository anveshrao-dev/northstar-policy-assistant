# Local controlled-live runtime

## Access boundary

The dedicated launcher starts the existing Streamlit application with an explicit
local access authorization. It requires Darwin and Python 3.12, rejects hosted
Replit markers, and binds only to `127.0.0.1`. CORS/XSRF protections remain enabled;
static serving, telemetry, TLS overrides and public binding are rejected or disabled.

This is a trusted single-user Mac policy, not remote authentication. Do not expose
it through tunnels, proxies, port forwarding or a shared multi-user host. Loopback
does not protect against another process controlled by the same OS user.

Each request verifies actual Streamlit configuration, a direct loopback WebSocket
client, exact numeric Host/Origin and absence of forwarded headers. Missing or
duplicate context fails closed. The pinned Streamlit client-context interface is
used because its public `ip_address` is `None` for loopback; upgrades require
regression review. Hosted operation retains independent Private Preview verification.

## Authorized execution and publication order

The controlled path is:

1. Explicit operator authorization and browser session consent.
2. Access verification and durable reservation under the authority lock.
3. Resource setup, verified retrieval and complete evidence assembly.
4. One authorized generator transport attempt and response validation.
5. Resource cleanup and durable settlement.
6. Durable typed public-result recording, then display.

Paid QA/graph setup rejects missing or mismatched permits before credential or
retrieval-resource access. Access, pricing and request binding are rechecked at
the required boundaries. A provider's `allow_paid` flag alone is insufficient;
the Nebius adapter requires an active durable single-use stage permit. The legacy
`answers ask --allow-paid` entry point cannot bypass that contract.

The LINEAR reservation covers one generator call, up to 30,720 input-bound and
2,048 output tokens. The optional controlled semantic graph reserves all seven
possible stages: 184,064 input-bound plus 20,736 output tokens. Actual prompt/output
acceptance may be stricter. These are reserved upper bounds, not reported billing
usage or a guarantee that evidence fits a prompt.

Source hashes, policy eligibility, model identity, siblings, context budget,
structured output, citations and quotations remain mandatory. Invalid responses
are withheld. A legitimate evidence-limited abstention is distinct from resource,
provider or validation failure. Valid references do not establish general semantic
correctness; see the [reviewed actor safeguard](Reviewed-Actor-Validation.md).

Duplicate IDs, interrupted calls, uncertain outcomes and persistence failures do
not authorize retries, replacement invocations or refunds. Settlement or result
recording failure withholds the answer even if inference completed. The development
profile retains the complete reservation for each attempt. Its allowance survives
sessions, reloads and process restarts.

## Mac development startup

Use the existing locked Python 3.12 environment with UI/LLM dependencies and pinned
MiniLM assets. Chroma must already be initialized and source-compatible. The
contained `.local/tmp`, `.local/cache` and `.venv/bin` paths must exist; the launcher
does not fall back to global temporary directories.

Prepare, independently review and activate a distinct authority using the
[development procedure](Local-Development-Authority.md). Then separately authorize
paid execution, select its approved plan/run binding, and launch:

```bash
bash scripts/run_local_live.sh --approve-local-live --development --backend chroma
```

Open `http://127.0.0.1:8501`, select Live Q&A, choose the permitted policy mode and
enable paid answers for that session. No question is submitted automatically.
The default Offline Demo and retrieval-only Policy Search remain separate.

Startup validates the authority and current pricing without initiating inference.
Credential and backend checks at reserved resource setup still apply; a healthy
server alone does not establish live-provider readiness. Startup never initializes
accounting, downloads models, builds an index, renews pricing or changes limits.

## Profile and authority isolation

Development mode is LINEAR/Chroma only. It rejects graph, Pinecone, batch, one-shot,
handoff and external-authority selectors. Generic controlled-local and supported
legacy one-shot/hosted profiles retain their separate approval contracts; there is
no fallback from a missing or spent one-shot to a larger batch.

An activated one-shot profile binds exact question bytes, invocation ID, mode,
reference date, backend, model and corpus. Duplicate clicks, another session or a
restart cannot recreate its permit. UI startup never retires a source authority or
activates a destination. Accounting is authoritative on one filesystem; Mac and
Replit copies would enforce independent spending limits. Migration or identity
recovery requires its separate reviewed mechanism, not a ledger copy.

The [architecture guide](../README.md#hosted-and-legacy-compatibility) summarizes
hosted and legacy boundaries. The code retains filesystem identities, private
permissions, exclusive locks, immutable approvals and fail-closed transition guards.

## Offline verification and operational limits

Synthetic tests cover missing/expired/mismatched approvals, loopback and hosted
separation, session consent, duplicates, resource ordering, concurrency, interrupts,
budget exhaustion, uncertain settlement, recording failure and malformed provider
responses. Graph tests cover bounded routing/grading/semantic transitions with
fake providers. They do not establish live graph quality or provider billing.

```bash
.venv/bin/python -B scripts/test_offline.py test_local_live.py
.venv/bin/python -B scripts/test_offline.py test_live_authorization.py
.venv/bin/python -B scripts/test_offline.py test_one_shot_ui.py
.venv/bin/python -B scripts/test_offline.py
.venv/bin/python -B scripts/validate_evaluations.py
bash -n scripts/run_local_live.sh
```

Run tests in a credential-free environment. The Python socket guard is not an OS
sandbox for arbitrary external programs. Cached-model integration is opt-in and
must use verified local assets without downloading. See [offline integrity](Offline-Integrity-Checks.md)
and [Chroma isolation](Secure-Chroma-Initialization.md).

Live Mac Standard RAG has been demonstrated; complete live semantic LangGraph has
not. Known timeout, response-envelope and semantic-attribution limitations are
summarized in the [architecture guide](../README.md#verification-and-limitations).
If an authority is guarded, corrupt, expired, identity-mismatched or exhausted,
stop. Never clear guards, restore older spending records or retry automatically.
