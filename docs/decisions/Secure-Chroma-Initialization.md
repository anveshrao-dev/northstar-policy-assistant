# Secure Chroma initialization

Chroma 1.5.9 constructs settings while importing its package, including reading a
working-directory `.env`. Supplying `_env_file=None` to a later client does not
prevent that earlier read. Northstar therefore imports Chroma only in a dedicated
local worker. Dependency versions and the embedding/index configuration are unchanged.

## Initialization boundary

Each locked Chroma adapter operation starts a fresh Python process in an empty,
owner-only temporary directory under the selected project's `.local/`. The worker
validates its canonical location, permissions, project/index paths and inherited
lock identity before importing Chroma. It receives an explicit minimal environment,
without credentials, provider endpoints, Chroma environment overrides, application
authorization flags or accounting selectors. The application process keeps its
existing working directory and environment.

The worker uses the existing explicit embedded Rust client, cosine collection,
disabled telemetry/reset and `embedding_function=None`. A worker-local guard
rejects dotenv/secret/accounting file access, cwd changes and Python network
connections. This is isolation from accidental dependency configuration, not an
operating-system sandbox against hostile code or another process with the same
operating-system privileges.

Communication uses an anonymous local socket pair, with no listener or network
endpoint. Only the existing collection operations are accepted. JSON frames are
bounded to 16 MiB, request identities must be sequential, and parent send/receive
phases have 60-second limits. A worker accepts at most 1,024 requests during its
single locked operation. Errors are terminal fixed categories; dependency
tracebacks and raw configuration are not returned or logged. There is no automatic
operation retry or fallback to an in-process client.

## Integrity and lifecycle

The parent retains policy eligibility, source and sibling validation, record/vector
digests, context budgets and atomic publication-sidecar handling. The worker verifies
the collection configuration and preserves the existing result contract.

The same exclusive index-lock descriptor is inherited by the worker. Normal cleanup
closes the channel, waits for exit and removes the temporary directory before the
parent closes its lock. A non-exiting worker is killed and reaped. Abrupt parent
termination leaves the lock held until the worker exits; a partial index write
retains the existing incomplete-publication semantics. No uncertain mutation is
automatically retried. Forced parent termination may leave a private temporary
directory; it is never reused as a trusted initialization directory.

## Application compatibility

Local and hosted launch commands are unchanged. Streamlit Offline Demo remains
independent of Chroma. Policy Search and Chroma-backed Q&A reach the worker through
the existing `ChromaStore` interface. Pinecone retrieval is unchanged. Provider,
budget, consent, loopback and Replit Private Preview checks run in the application
process as before; the worker has no paid-inference role or authority.

Supported application paths must use the adapter. Direct third-party `import
chromadb` calls bypass this boundary and still have the dependency's original
dotenv behavior. Application code must not add such imports outside the worker.
The implementation uses Unix file-descriptor passing and `flock`, consistent with
the existing macOS/Linux adapter. Windows is not supported by this path.

## Verification and trade-offs

Synthetic subprocess tests cover conflicting, malformed and absent dotenv files;
real persisted retrieval and approval filtering; Streamlit AppTest startup and
Policy Search; unchanged parent cwd/environment; fixed failure responses; worker
crashes, interruptions, framing limits, lock retention and cleanup. Existing tests
cover source integrity, corrupt indexes, hosted launchers and provider authorization.
No real accounting or provider requests are needed for these checks.

Run the focused checks with the installed development environment:

```sh
.venv/bin/python -B scripts/test_offline.py test_chroma_isolation.py
.venv/bin/python -B scripts/test_offline.py test_vector_store.py
.venv/bin/python -B scripts/test_offline.py test_policy_search.py
```

Fresh process startup adds latency to each locked Chroma operation, including
status checks. Historical retrieval latency measurements describe the earlier
runtime and have not been replaced or remeasured here. Reusing workers would need
a separately reviewed lifecycle and concurrency design. The current corpus has
189 embedding records. Full-record integrity reads must fit the 16 MiB frame cap;
larger future corpora may fail closed and need bounded pagination before use.
An over-limit write verification retains an incomplete index, never a partial
ready publication. Linux/Replit execution
and actual browser layout are not established by Mac subprocess/AppTest coverage.

See [offline validation](Offline-Integrity-Checks.md) for the full regression and
limited CI distinction, and [local controlled live](Local-Controlled-Live-Runtime.md)
for the separate authorization requirements.
