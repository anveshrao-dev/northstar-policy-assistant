# Source-reviewed actor attribution

Citation membership and exact quotations establish provenance, not whether a
generated claim assigns an action to the correct actor. Standard RAG retains
this limitation; the optional graph's model-based semantic checks are a separate
mechanism and are not added to linear requests.

The shared answer validator now applies a deterministic rule to claims and
resolved obligation statements citing the reviewed **HR-001 v1.0 §4.1 approved
synthetic snapshot**. That source requires receiving-manager approval and HR
recording before the transfer effective date. The rule binds document, version,
status, section and source hash, and requires the full, unambiguous initiation
statement in the supplied eligible context.

Accepted wording consists of entire supplied source spans or complete sentences
from a small explicit reviewed vocabulary in
[reviewed_actors.py](../../src/northstar_atlas/reviewed_actors.py). Only whitespace
is normalized. Correct source text and reviewed multi-role paraphrases are
preserved verbatim. Actor substitutions, ambiguous pronouns, negations, appended
unsupported assertions and other unreviewed wording are withheld through the
existing `response_rejected` path. Generated answers are never repaired silently.

This conservative rule can reject a correct paraphrase that has not been
reviewed. It does not parse arbitrary English or certify answer completeness.
It does not apply the approved rule to proposed drafts, changed snapshots or
other sections. Claims citing only other sections, uncited clarification fields
and unrelated obligations still require semantic review. A future source change
requires a fresh rule review; no general assumption that HR never approves
anything is made. Citation/quotation checks and policy eligibility remain intact.

The rule adds no prompt fields, provider parameters, model calls or retries.
Linear and graph generation share the same deterministic rejection boundary.
A returned provider call followed by answer rejection retains its full reserved
attempt and estimated cost; provider uncertainty remains a separate outcome.
The caller's durable settlement and result-recording sequence is unchanged.

Offline tests in [test_actor_obligations.py](../../tests/test_actor_obligations.py)
use genuine synthetic policy evidence, authored responses and temporary ledgers.
They cover correct and incorrect roles, scope/status boundaries, incomplete or
ambiguous evidence, graph withholding, and durable accounting after rejection.
These are deterministic regression checks, not a new live evaluation or a
change to frozen retrieval measurements.
