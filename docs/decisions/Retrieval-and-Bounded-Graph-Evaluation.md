# Retrieval and Bounded-Graph Evaluation

**Date:** 2026-10-24.

**Evidence:** Historical retrieval measurements and offline graph checks; not a live answer-accuracy benchmark.

**Scope:** Retrieval strategies and the original optional `bounded_v1` LangGraph
profile as of 2026-10-24. No application changes or hosted inference. This report
does not evaluate the later LLM routing, grading or semantic-checking profiles.

## 1. Findings and interpretation

Hybrid has the best expanded evidence recall, **70.44%**, versus **64.67%** dense and **66.78%** BM25. All three fully cover only **6/15** frozen questions. Every strategy's top-five order and evidence metrics match the earlier hybrid experiment. No retrieval regression was observed on this snapshot.

The actual graph enforces the tested call bounds, rejects malformed or invalidly cited responses, and preserves source eligibility. Reviewed Terraform requirements and the evidence-backed retention limitation pass with real local retrieval and deterministic fake generation. These are offline contract results, not new Nebius results.

Two important limitations remain:

- Nonempty evidence is sufficient for ordinary-query routing, even when cross-policy evidence is missing. Such queries retain `coverage.status = not_assessed`; the graph does not automatically establish relevance, completeness, ambiguity, or unsupportedness.
- An adversarial fake claim inventing an approved retention duration, paired with a genuine quotation from the proposed Compliance policy, **passed ordinary-query structural validation**. The evaluator rejects its semantic assessment. Citation membership and quotation provenance do not enforce entailment or prevent contradictory free-form wording. This is a material limitation for unrestricted policy guidance, not a successful answer.

These measurements support comparison of retrieval configurations. They do
**not** establish readiness for unrestricted authoritative enterprise advice or
production release. The findings support retaining the linear
QAService/Streamlit defaults while evaluating hybrid and graph behavior explicitly.
No deployment was performed as part of this evaluation. Later implementation and
live observations are documented separately in the [engineering guide](../README.md).

## 2. Frozen inputs and provenance

Inputs include the [frozen dataset and methodology](../../evaluations/README.md),
the dense and backend experiments preserved in Appendices A and B, and the public
[evaluation implementation](../../src/northstar_atlas/evaluation/) and
[graph contracts](../README.md#optional-langgraph-profiles).

| Item | Evaluated value |
| --- | --- |
| Dataset | `northstar-atlas-m1.8`, version 0.1; 15 questions; categories A/B/C/D = 3/6/3/3 |
| Required evidence | 40 question/reference occurrences, 31 distinct document/section pairs |
| Source snapshot | Six manifest-verified policies; 151 canonical chunks; 189 refined records |
| Eligibility reference date | **2026-10-22**, deliberately retained for baseline comparability |
| Approved-only evidence | HR-001 v1.0, Approved (Synthetic), effective 2026-10-01 |
| Draft review | HR plus five v0.1 Draft — Proposed (Synthetic) policies |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Vectors | 384 dimensions, normalized; cosine distance for Chroma |
| Local runtime | Python 3.12.15, Intel macOS; Chroma 1.5.9, LangGraph 1.2.16, SDK 0.4.9 |
| Ranking configuration | Dense top 5; BM25 top 5; hybrid dense 20 + BM25 20, RRF constant 60, final top 5 |

Exact identities:

```text
questions.json SHA-256:
e33b1dbccd7e8df82651f8a58f7044b3e2c3032cce1a82686bc792f2c7a241b4
corpus_manifest.json SHA-256:
1c6efd57db22e900da7118eca62f8f075d96e5b846c1a30e3889628cd23860b1
corpus fingerprint:
20f5ec1d95989e7c6f6b12579d089dea5be0fe815fd5ba357902d195d49dfac9
immutable local snapshot ID:
e55dcdeca17453b27d45f4d3663ef72b9ba3e635f43689899a8d66b52cf27860
```

The ready 189-record Chroma receipt, corpus, configuration and frozen local snapshot were checked before and after evaluation. The existing comparison adapter checked actual selected text, metadata and vector direction against that snapshot; the retrieval experiment touched 149 distinct selected records through normal searches/context expansion. It did not rebuild the index or run a separate full-record audit. Database internal housekeeping bytes were not used as a reproducibility assertion; receipt identity, count and verified payloads were.

The local snapshot lives under `.local/pinecone/snapshots/`, but reading it invokes no Pinecone service. Pinecone was **not queried**. Its earlier measurements remain historical evidence only. Credentials were not loaded. Existing model files were used offline; no document embeddings were regenerated.

## 3. Scoring definitions and experiment boundaries

The unchanged evaluator uses `complete-source-section-v1`, the same deterministic proxy as M2.5/M3.4/M5.2:

- A unit is a document-qualified section. Coverage requires its canonical source spans, not merely its heading or a cross-reference to it. A parent reference requires the parent's own content and all dot-delimited descendant subsections. A subsection alone does not satisfy the parent.
- **Recall@5:** covered required units divided by all required units, macro-averaged across questions.
- **Hit@5:** whether at least one required unit is covered.
- **Full Evidence Coverage@5:** whether every required unit is covered.
- **Reciprocal Rank@5 / MRR@5:** reciprocal of the earliest direct-rank prefix covering a required unit, then averaged. Expanded scoring uses the sibling closure of that prefix; siblings never receive independent direct ranks or votes.
- Duplicate section references earn no repeated credit. Direct scoring uses only the five ranked chunks; expanded scoring adds their required, eligible sibling context.

Exact source-span coverage is a conservative proxy, not a semantic judgment. It can penalize a missing paragraph that is irrelevant to a particular question. Conversely, copying a whole section does not prove an answer interprets it correctly. Cosine similarity, BM25 scores, and RRF scores are neither answer confidence nor interchangeable scales.

Categories C and D are scored for evidence supporting clarification or corpus limitations, **not** for ordinary answerability. Overall scores include their evidence needs; answerable A+B scores are reported separately. The graph experiments do not feed expected answers, required evidence, or category labels into ordinary runtime routing. Explicit reviewed fixtures are identified separately.

## 4. Retrieval results

All percentages below are macro-averages. MRR is on a 0–1 scale.

| Strategy | Evidence | Recall@5 | Hit@5 | Full coverage@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | Direct | 55.22% | 86.67% | 26.67% | 0.6944 |
| Dense | Expanded | 64.67% | 93.33% | 40.00% | 0.8667 |
| BM25 | Direct | 51.56% | 80.00% | 26.67% | 0.6133 |
| BM25 | Expanded | 66.78% | 93.33% | 40.00% | 0.8333 |
| Hybrid | Direct | 57.44% | 86.67% | 26.67% | 0.6556 |
| Hybrid | Expanded | 70.44% | 100.00% | 40.00% | 0.8833 |

Sibling expansion improves recall by 9.44, 15.22 and 13.00 percentage points respectively. Hybrid's 100% expanded Hit@5 means each question has at least one covered unit; it does **not** mean all questions can be answered.

For the nine answerable A+B questions, direct/expanded recall is dense **67.96% / 80.00%**, BM25 **63.70% / 81.67%**, hybrid **67.96% / 82.22%**. All three have direct full coverage 3/9, expanded full coverage 5/9, and Hit@5 100%. Answerable expanded MRR is 1.0000, 0.8889, 0.9444 respectively.

### 4.1 Category breakdown

Each cell is **direct / expanded**. Categories: A precise HR (3), B cross-policy (6), C ambiguous (3), D unsupported requested values/procedures (3).

| Strategy | Category | Recall % | Hit % | Full coverage % | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | A | 100 / 100 | 100 / 100 | 100 / 100 | 1.0000 / 1.0000 |
| BM25 | A | 100 / 100 | 100 / 100 | 100 / 100 | 0.8333 / 0.8333 |
| Hybrid | A | 100 / 100 | 100 / 100 | 100 / 100 | 0.8333 / 0.8333 |
| Dense | B | 51.94 / 70.00 | 100 / 100 | 0 / 33.33 | 0.7639 / 1.0000 |
| BM25 | B | 45.56 / 72.50 | 100 / 100 | 0 / 33.33 | 0.7000 / 0.9167 |
| Hybrid | B | 51.94 / 73.33 | 100 / 100 | 0 / 33.33 | 0.7639 / 1.0000 |
| Dense | C | 22.22 / 33.33 | 66.67 / 100 | 0 / 0 | 0.4444 / 0.8333 |
| BM25 | C | 33.33 / 44.44 | 66.67 / 100 | 0 / 0 | 0.5000 / 1.0000 |
| Hybrid | C | 33.33 / 44.44 | 66.67 / 100 | 0 / 0 | 0.5000 / 1.0000 |
| Dense | D | 50.00 / 50.00 | 66.67 / 66.67 | 33.33 / 33.33 | 0.5000 / 0.5000 |
| BM25 | D | 33.33 / 44.44 | 33.33 / 66.67 | 33.33 / 33.33 | 0.3333 / 0.5000 |
| Hybrid | D | 50.00 / 61.11 | 66.67 / 100 | 33.33 / 33.33 | 0.4167 / 0.5833 |

### 4.2 Missing expanded evidence by question

These are exact required units still missing under the unchanged source-coverage proxy. A dash means none. The historical machine-readable result additionally retained direct misses, partial spans, ranks and scores; it is not distributed with this public report.

| Question | Dense | BM25 | Hybrid |
| --- | --- | --- | --- |
| 001–003: HR lifecycle | — | — | — |
| 004: transfer to SRE | HR-001 §§4.3, 4.4; SEC-001 §§5.2, 4.1 | HR-001 §4.3; SEC-001 §4.1 | HR-001 §§4.3, 4.4; SEC-001 §4.1 |
| 005: Terraform regeneration | — | — | — |
| 006: Jenkins ownership | —; prompt blocked | —; prompt blocked | —; prompt blocked |
| 007: AKS identity / Key Vault | IT-001 §5.2 | IT-001 §5.2; SEC-001 §§6.1, 7.1 | IT-001 §5.2; SEC-001 §7.1 |
| 008: failed deployment / incident | SRE-001 §§1.1, 3.2 | SRE-001 §1.1 | SRE-001 §1.1 |
| 009: certificate disruption | SRE-001 §5.2 | SRE-001 §5.2 | SRE-001 §5.2 |
| 010: unspecified Jenkins failure | SEC-001 §7.2; SRE-001 §1.1 | Same | Same; prompt blocked |
| 011: vague production access | IT-001 §5.1; SEC-001 §3.1 | Same | Same |
| 012: extended deployment | PLAT-001 §4.3; SRE-001 §7.2 | PLAT-001 §4.3 | PLAT-001 §4.3 |
| 013: approved retention duration | — | — | — |
| 014: exact MFA / rotation value | SEC-001 §8.2 | SEC-001 §§4.1, 8.2 | SEC-001 §8.2 |
| 015: approved GCP procedure | PLAT-001 §§1.1, 2; SEC-001 §6.1 | PLAT-001 §1.1; SEC-001 §6.1 | Same as BM25 |

Representative successes are the HR lifecycle questions, expanded Terraform conditions/evidence, and the retention policy gap. Representative weaknesses are cross-policy transfer/removal gates, missing workload/secret authorization evidence, and conditional scope/escalation evidence. No ordinary answer is certified by these retrieval successes.

### 4.3 Historical comparison

All six aggregate metric rows and all 45 question/strategy top-five orders match M5.2. Dense metrics also exactly match M2.5 and the Chroma side of M3.4. No tuning or dataset changes were made.

M3.4's historical Pinecone evidence coverage equals Chroma's, with direct MRR 0.6911 and expanded MRR 0.8333 versus Chroma 0.6944/0.8667. That earlier backend comparison does not establish current Pinecone readiness, latency or usage. No managed-backend check was needed or performed here.

## 5. Latency, context size and budget failures

Warm retrieval measurements use three passes, rotating strategy order, 45 samples per strategy. They include readiness checks, selected payload verification and sibling fetching; exclude one-time query encoding, evidence scoring, context preparation and file IO. Hybrid includes dense20, BM25-20 and fusion. Frozen ranks remained identical across passes. The existing adapter returns ranked hits and siblings together, so there is one measured retrieval latency per strategy, not a separately instrumented direct-only latency.

| Strategy | Mean ms | Median ms | Maximum ms |
| --- | ---: | ---: | ---: |
| Dense | 18.94 | 18.12 | 24.61 |
| BM25 | 7.50 | 7.91 | 10.98 |
| Hybrid | 40.90 | 41.39 | 78.30 |

The first correctness pass averaged 20.22/7.94/40.46 ms respectively. These are local observations, not service-level guarantees. BM25 construction took 8.46 ms. Cached MiniLM loading took 26.00 s; sixteen distinct query encodings were reused across all retrieval arms and timing passes. Query encoding mean/median/max was 358.81/14.26/5,524.88 ms, dominated by the first observed cold query. Total initial setup took 37.72 s. Graph evaluation was a separate run with cached-model query encoding; no document encoding occurred. No memory benchmark or Linux performance measurement was performed.

Context sizes below are before full-prompt acceptance. Expanded character counts include chunk context duplication; the artifact also records deduplicated source characters and exact spans.

| Strategy | Direct chars min / median / max | Expanded chars min / median / max | Expanded chunks min / median / max | Evidence bound mean / median / max |
| --- | --- | --- | --- | --- |
| Dense | 1,231 / 4,550 / 6,052 | 1,231 / 4,986 / 17,338 | 5 / 6 / 15 | 13,369 / 11,170 / 26,245 |
| BM25 | 1,185 / 4,973 / 6,076 | 1,185 / 5,741 / 18,532 | 5 / 6 / 16 | 13,922 / 12,122 / 28,701 |
| Hybrid | 1,185 / 4,549 / 6,076 | 1,185 / 6,183 / 17,731 | 5 / 7 / 15 | 14,160 / 13,104 / 26,592 |

The existing Qwen UTF-8/NFC byte bound is deliberately conservative; these numbers are **not actual tokenizer usage or billing tokens**. Default configured context is 32,768, output reservation 1,024, margin 512. Evidence-only budget is 31,232. No evidence group was excluded and no context warning occurred during the retrieval comparison. Nevertheless, instructions, question and structured-response schema overhead make four complete prompts too large:

| Question / strategy | Input bound including margin | Plus output reservation | Over 32,768 by |
| --- | ---: | ---: | ---: |
| EVAL-006 dense | 33,686 | 34,710 | 1,942 |
| EVAL-006 BM25 | 36,142 | 37,166 | 4,398 |
| EVAL-006 hybrid | 34,033 | 35,057 | 2,289 |
| EVAL-010 hybrid | 33,485 | 34,509 | 1,741 |

For these rows the diagnostic bound was calculated offline with a larger temporary validation ceiling solely to expose the number; the real evaluation retained the default ceiling and rejected generation. No LLM was called and no critical group was silently trimmed. Successful evidence assembly and an empty warnings list therefore do not imply a sendable prompt.

Full-prompt-ready counts are **14/15 dense, 14/15 BM25, 13/15 hybrid**. Counting blocked requests as delivering no evidence reduces expanded recall to **58.00%, 60.11%, 61.56%** respectively. This separate application-delivery diagnostic does not replace or redefine frozen retrieval Recall@5. Hybrid's evidence gain is partly offset by context cost.

## 6. Actual LangGraph evaluation

### 6.1 Real retrieval with fake generation on all frozen questions

Ran `GraphQAService` against actual local dense, BM25 and hybrid adapters for **45 question/strategy runs**. Each initial strategy had one alternate configured (dense→BM25, BM25→dense, hybrid→BM25), using unchanged question text, mode, date and top-k. The catalog revalidates eligible source identity and siblings. Generation used a deterministic fake that copies one contiguous retrieved paragraph with its real citation/quotation; it receives no expected answer, evidence rubric or category label.

This deliberately simple fake checks integration and structural publication, **not whether the returned paragraph answers the question**. It does not simulate intelligent ambiguity detection or abstention selection.

| Initial strategy | Structurally accepted fake `answer` | Context errors before generation | Retrieval attempts per question | Fake provider calls total |
| --- | ---: | ---: | --- | ---: |
| Dense | 14 | 1: EVAL-006 | 1 | 14 |
| BM25 | 14 | 1: EVAL-006 | 1 | 14 |
| Hybrid | 13 | 2: EVAL-006, EVAL-010 | 1 | 13 |

Every accepted ordinary result reports `not_assessed`. Accepted citations in approved-only runs are HR-001 only; proposed labels remain on draft citations. There are no source/citation integrity violations. Success follows `router → retriever → assessor → generator → validator`; prompt overflow follows `router → retriever → assessor → validator`, with zero provider calls.

No ordinary run took an alternate because retrieval was nonempty; partial required-evidence coverage is not automatically recognized as insufficiency. The C/D paragraph fixtures can be structurally accepted as `answer` even though the frozen behavioral expectation is `clarify`/`abstain`. This is **not** a measured real-model behavior failure or behavioral success rate. It demonstrates that the graph itself does not infer those decisions from the benchmark or enforce semantic relevance.

### 6.2 Controlled routes, reviewed requirements and fault injection

Twenty-four labeled scenarios used the real graph and existing validators. Most inject catalog-derived source fixtures or controlled errors; two reviewed cases use actual local retrieval. All assertions about expected behavior/call counts passed. The final adversarial case intentionally exposes a semantic weakness and must not be counted as a safe policy answer.

| Scenario | Observed result | Searches / fake inference calls |
| --- | --- | --- |
| Approved HR / proposed Compliance fixture | Validated source-extractive `answer`, correct labels | 1 / 1 each |
| Empty dense, relevant BM25 | Alternate succeeds; prior hits replaced | 2 / 1 |
| Two empty attempts | `abstain`, insufficient supplied evidence, no fabricated citation | 2 / 0 |
| Single-strategy empty plan | `abstain` after configured single attempt | 1 / 0 |
| Trusted question-scoped clarification | `clarify` without retrieval/provider | 0 / 0 |
| Fake generated clarification | Validated `clarify` | 1 / 1 |
| Fake no-evidence abstention | Validated `abstain`; no unsupported policy claims | 1 / 1 |
| Required reviewed source missing first, present second | Alternate succeeds; no preflight evidence injection | 2 / 1 |
| Required reviewed source absent twice | `context_failed`; cannot bypass obligations with generic abstention | 2 / 0 |
| Four Terraform obligations with real retrieval | `answer`, all application-owned source spans rendered, proposed citation | 1 / 1 |
| Retention limitation with real retrieval | Cited evidence-backed `abstain`, documented-gap basis | 1 / 1 |
| Missing required stop/amend obligation | `response_rejected`, no candidate published | 1 / 1 |
| Fabricated citation / quotation | `response_rejected` | 1 / 1 each |
| Malformed JSON / `finish_reason=length` | `response_rejected` | 1 / 1 each |
| Provider timeout | `provider_failed`, no retry or policy abstention | 1 / 1 |
| Draft leak in approved-only mode | `context_failed` before generation | 1 / 0 |
| Missing sibling / stale source hash | `context_failed` before generation | 1 / 0 each |
| Small context budget | `context_failed`, no silent evidence loss | 1 / 0 |
| Fabricated reviewed source statement | `configuration_failed` before retrieval | 0 / 0 |
| Unsupported claim with genuine quote, no reviewed obligations | Structurally accepted; **semantic assessment fails** | 1 / 1 |

Alternate-success trace: `router → retriever → assessor → retriever → assessor → generator → validator`. Exhaustion omits generator. Trusted clarification and invalid reviewed input take `router → validator`. Invalid retrieved payloads take `router → retriever → validator`. Invalid generated candidates reach validator as failures and expose only safe public errors; no rejected sentinel text appeared in results.

A three-strategy configuration is rejected before execution. No tested invocation exceeded two retrieval attempts or one provider call. Deterministic fixtures also confirm the linear QAService and graph return identical public contracts for identical approved evidence. The complete existing suite supplies broader router, backend, retrieval-failure, budget, grounding, UI and compatibility regressions.

### 6.3 Reviewed versus ordinary completeness

The Terraform diagnostic uses the existing four caller-reviewed obligations and exact source spans, with the application-owned extractive profile. All four IDs are covered, quoted source matching permits only whitespace normalization, and the rendered citation remains `PLAT-001 v0.1 §6.1 (proposed)`. Input bound is 22,224 before the 1,024 output reservation. The source statements preserve renewed technical approval, no automatic additional Change Management approval solely for regeneration, authorized scope/risk/window/conditions, and stop/amend/escalate boundaries. Source reproduction is established; general semantic completeness is not.

The retention diagnostic uses the existing evidence-linked limitation and restricted missing-information label. It returns `abstain`, cites `COMP-001 v0.1 §11.1 (proposed)`, and quotes the exact statement that no numeric period is set. Input bound is 19,980. It establishes a documented gap in supplied proposed evidence, not the nonexistence of a rule elsewhere. Coverage is `required_entries_present`, with `no_period_established` evidence-linked; its authored semantic interpretation is assessed separately from structural validation.

In the adversarial probe, an invented approved numeric duration was accompanied by a genuine COMP-001 quotation. The ordinary validator accepted membership/provenance, while coverage stayed `not_assessed`; the free-form claim contradicted the source and its proposed authority. No real model produced this fixture and it is not published here as policy guidance. The result confirms that existing semantic limitations are observable at the public-result boundary, not eliminated by adding a graph. Changing validators or the successful reviewed paths was outside this evaluation's scope.

## 7. Supplementary AKS temporary-access diagnostic

Exact question, kept outside the frozen dataset:

> An engineer needs temporary production access to troubleshoot an AKS issue. Who should approve it, and which team grants the access?

Mode: `draft_review`. The existing supplementary rubric has five references: IT-001 §§10.1, 9.1; SEC-001 §§9.1, 5.2, 6.1. It is a diagnostic rubric, not a complete executable authorization decision.

| Strategy | Direct / expanded coverage | Direct top-five section sequence |
| --- | --- | --- |
| Dense | 2/5 / 2/5 | IT-001 §10.1; SEC-001 §9.1; SEC-001 §3.1; PLAT-001 §14; another SEC-001 §3.1 chunk |
| BM25 | 1/5 / 1/5 | SEC-001 §3.2; SEC-001 §9.2; SEC-001 §9.1; COMP-001 §10.1; IT-001 §6.2 |
| Hybrid | 2/5 / 2/5 | SEC-001 §9.1; SRE-001 §2; IT-001 §10.1; COMP-001 §6.1; SEC-001 §3.2 |

All three miss the complete technical-provisioning/decision references in final context. Dense sees a SEC-001 §5.2 candidate at rank 9; BM25 sees SEC-001 §6.1 at rank 16; IT-001 §9.1 is absent from both top-20 candidate lists. Fusion does not promote those candidates into final top five. This is observed candidate/rank evidence; it does not isolate a model-training or tokenizer root cause.

Source inspection, **not retrieval credit**, establishes the proposed boundaries:

- [IT-001 §10.1](../../data/policies/IT-001.md) routes production AKS requests for Security assessment of actual scope; production access is not automatically administrative or routine development access.
- [SEC-001 §§5.2, 9.1](../../data/policies/SEC-001.md) separate business approval, Security authorization and technical fulfillment; temporary access needs applicable approvals, approved start/end, monitoring, and revocation ownership. Missing decisions or delegation block provisioning.
- IT-001 §9.1 assigns workforce identity/authorized standard group work to IT, and platform-owned mappings/grants/removals to Platform Engineering. SEC-001 §6.1 assigns cloud implementation within delegation to Cloud Platform and distinguishes Azure permissions from Kubernetes RBAC.
- SRE-001 §2 in hybrid context provides a broad ownership table, but does not replace the detailed IAM/cloud grant and delegation conditions. A temporary troubleshooting request is not automatically an emergency; BM25's emergency hit does not establish that classification.

The evidence partially covers Security authorization, but does not adequately supply both IAM identity responsibilities and the conditional Cloud Platform executor. The owning resource, requested permissions, business approver/delegation, applicable change controls and expiry decisions still need case-specific confirmation. All relevant policies here remain proposed. Do not label this question complete or infer a blanket grant by Security, IT or SRE. Sibling expansion protects selected section families; it does not automatically traverse every cross-policy referral.

## 8. Failure analysis and future work

| Failure class | Evidence | Implication |
| --- | --- | --- |
| Retrieval omission | EVAL-004, 007–012, 014–015 and supplementary AKS lack required cross-policy units | Better candidate coverage/selection is needed; a citation cannot repair a missing source |
| Fusion tradeoff | Hybrid improves aggregate recall but loses some dense-only EVAL-007 evidence and adds EVAL-010 prompt overflow | No strategy dominates every question; preserve per-question results |
| Context failure | EVAL-006 all arms, EVAL-010 hybrid | Complete evidence may be retrieved but unavailable to generation under current conservative budget |
| Ordinary sufficiency limitation | Nonempty partial evidence routes directly to generation | Bounded retries do not recover omissions unless insufficiency is actually detected or reviewed requirements are supplied |
| Generation contract failure | Missing obligation, invented citation/quote, malformed/truncated output | Safely rejected with no retry and no published candidate |
| Semantic failure | Adversarial ordinary claim with genuine support quote passes structure | Membership/provenance is not entailment; unsupported authority and values need separate semantic assessment |
| Operational failure | Injected provider timeout remains `provider_failed` | Do not disguise infrastructure failure as policy abstention |

Follow-up evaluation directions (not results of this historical experiment):

1. Expand independently reviewed semantic/obligation evaluation, including negative claims, draft-authority contradictions and conditional approval gates. Keep ordinary completeness explicitly unassessed until there is evidence for a stronger claim.
2. Investigate bounded multi-part retrieval or explicit cross-policy evidence requirements for ownership questions; keep source authority and delegation gates intact. Do not inject the frozen answer key into runtime retrieval.
3. Test candidate selection/reranking and context allocation against a separate diagnostic set. Compare per-question evidence gains, latency and full-prompt delivery, rather than aggregate recall alone.
4. Measure a verified generation tokenizer or carefully reviewed prompt/schema overhead reduction to recover budget-blocked cases. Do not drop sibling conditions or increase limits silently.
5. Use the frozen baseline for regression, add held-out questions without rewriting it, and conduct a separately approved live answer evaluation. Fake generation cannot establish model behavior, faithfulness, refusal reliability, pricing or hosted latency.

The small development dataset has been inspected repeatedly and is not a held-out estimate of enterprise performance. There are no statistical confidence claims. No reranker, alternative model, graph LLM planner, new cloud component or deployment change was introduced.

## 9. Reproducibility and evidence boundaries

The historical evaluation recorded **527 passing tests, zero failures/errors/skips,
115.75 seconds**, with cached-model checks enabled. Its additional checks covered
45 graph/retrieval runs and 24 controlled scenarios. These historical counts are
not the current regression-suite size and do not certify the adversarial semantic
fixture as a valid policy answer.

The public dataset, corpus, scoring functions and deterministic regression tests
are retained. Historical machine-readable runs, frozen local vector snapshots and
ad hoc timing harnesses are not included. Accordingly this report preserves the
recorded measurements but does not promise byte-identical replay or timings from
a fresh clone. Any new benchmark must identify its own assets, environment and
results rather than overwrite this baseline.

With the locked environment already installed, use the guarded offline suite and
the frozen integrity validator:

```bash
mkdir -p .local/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
.venv/bin/python -B scripts/test_offline.py
.venv/bin/python -B scripts/validate_evaluations.py
```

Cached-model checks require separately provisioned pinned assets and explicit
opt-in. No command here authorizes a provider call or reconstructs private results.
See [offline integrity](Offline-Integrity-Checks.md) for CI scope, and the
[architecture guide](../README.md#verification-and-limitations) for later live
observations and the narrow actor safeguard. Later changes do not retroactively
alter the historical measurements below.

## Appendix A — Dense retrieval baseline

Historical local dense experiment, 2026-10-22 reference date. The following results
and caveats are retained from the original dense baseline. Its timings include
query encoding and differ from the later adapter-only measurements.

### A.4 Overall Results

| Metric, all 15 questions | Direct top 5 | Top 5 + required siblings |
| --- | ---: | ---: |
| Macro Recall@5 | 55.22% | 64.67% |
| Hit@5 | 86.67% (13/15) | 93.33% (14/15) |
| Full Evidence Coverage@5 | 26.67% (4/15) | 40.00% (6/15) |
| MRR@5 / expansion-trigger MRR | 0.6944 | 0.8667 |
| Authority leakage / validation violations | 0 / 0 | 0 / 0 |

For the nine answerable A/B questions alone, recall is **67.96% → 80.00%**,
Hit@5 is **100% → 100%**, full coverage is **3/9 → 5/9**, and MRR is
**0.8426 → 1.0000**. None of these figures measures answer correctness.

Mean retrieval latency was **0.407709 s**. The slowest was EVAL-001 at
**3.914559 s**, the first inference query; the remaining 14 averaged **0.157220 s**.
Model loading took **12.979454 s** and total model/refinement setup **13.385917 s**,
excluded from retrieval timing. Each retrieval time includes source/integrity
validation, query embedding, Chroma search, and sibling fetching. It is not a
database-only benchmark. Scoring, token accounting, and output writing are excluded.
There was no benchmark-query warmup and only one measured pass per artifact;
latency is an observation, not a service-level guarantee.

Mean context grew from **791.27 to 1,296.13 MiniLM tokens** (+63.8%). Counts sum
each chunk including its special tokens and duplicated context; they are not a
future generation model's prompt-token count. EVAL-006 expanded from 5 to 15
unique chunks and from 1,021 to 3,322 tokens. Expansion improves completeness at
a measurable context cost; it does not recover arbitrary missing sections.

### A.5 Results by Category and Question

Each pair below is direct → expanded. All figures use the coverage proxy above.

| Category | N | Recall@5 | Hit@5 | Full coverage | MRR / trigger MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: precise HR | 3 | 100% → 100% | 100% → 100% | 100% → 100% | 1.0000 → 1.0000 |
| B: multiple documents | 6 | 51.94% → 70.00% | 100% → 100% | 0% → 33.33% | 0.7639 → 1.0000 |
| C: ambiguous | 3 | 22.22% → 33.33% | 66.67% → 100% | 0% → 0% | 0.4444 → 0.8333 |
| D: unsupported | 3 | 50% → 50% | 66.67% → 66.67% | 33.33% → 33.33% | 0.5000 → 0.5000 |

The fractions below are covered/required units, making each per-question recall
explicit. Missing means absent or incomplete under the declared proxy, not proof
that no useful information was retrieved. Direct-only partial units appear in
the last column; the historical JSON additionally contained both missing lists and every ranked result and is not included here.

| Question | Direct → expanded | Still missing after expansion | Direct-only partial units |
| --- | --- | --- | --- |
| EVAL-001 | 1/1 → 1/1 | None | None |
| EVAL-002 | 1/1 → 1/1 | None | None |
| EVAL-003 | 1/1 → 1/1 | None | None |
| EVAL-004 | 1/5 → 1/5 | HR-001 §§4.3, 4.4; SEC-001 §§5.2, 4.1 | None |
| EVAL-005 | 2/3 → 3/3 | None | COMP-001 §8.2 |
| EVAL-006 | 1/2 → 2/2 | None | SRE-001 §6.1 |
| EVAL-007 | 3/4 → 3/4 | IT-001 §5.2 | None |
| EVAL-008 | 2/4 → 2/4 | SRE-001 §§1.1, 3.2 | None |
| EVAL-009 | 2/4 → 3/4 | SRE-001 §5.2 | SRE-001 §6.1 |
| EVAL-010 | 0/3 → 1/3 | SEC-001 §7.2; SRE-001 §1.1 | SRE-001 §6.1 |
| EVAL-011 | 1/3 → 1/3 | IT-001 §5.1; SEC-001 §3.1 | None |
| EVAL-012 | 1/3 → 1/3 | PLAT-001 §4.3; SRE-001 §7.2 | None |
| EVAL-013 | 1/1 → 1/1 | None | None |
| EVAL-014 | 1/2 → 1/2 | SEC-001 §8.2 | None |
| EVAL-015 | 0/3 → 0/3 | PLAT-001 §§1.1, 2; SEC-001 §6.1 | None |

#### Ambiguous and unsupported cases: inspection of returned context

- **EVAL-010:** PLAT-001 §9.3 at rank 1 and the Jenkins row of SRE-001 §6.1 at
  rank 2 give conditional cloud/on-premises/application ownership and production
  impact distinctions. These are useful despite direct proxy recall of zero;
  the full SRE table needs siblings. Missing SEC-001 §7.2 and SRE-001 §1.1 leave
  labelled identity/incident-scope evidence incomplete. Hosting, failing component,
  and impact still need clarification by a future answer component.
- **EVAL-011:** IT-001 §5.2 at rank 1 separates business approval, Security
  approval, and technical provisioning, and blocks unresolved classification or
  controls. Missing request-detail and classification sections limit the evidence
  available for asking which production resource, permissions, and scope are meant.
- **EVAL-012:** PLAT-001 §11.3 at ranks 1 and 3 provides the recorded-authority and
  window-extension conditions: Rapid Response coordination is not permission to
  continue. Missing PLAT-001 §4.3 and SRE-001 §7.2 leave the required scope and
  escalation boundaries incomplete. SRE-001 §7.1 is useful returned context, but
  is not an adjudicated substitute for §7.2.
- **EVAL-013:** COMP-001 §11.1 at rank 1 explicitly sets no numeric retention
  period and distinguishes annual policy review from retention duration. This
  is evidence of a limit, not evidence of an approved number of years.
- **EVAL-014:** SEC-001 §4.1 at rank 2 says exact MFA methods require reviewed
  adoption. SRE-001 §10.2 and IT-001 §8.2 also say they do not invent rotation
  intervals, but the required SEC-001 §8.2 is missing. The dataset has no approved
  alternative labels, so these useful caveats do not replace that required unit.
- **EVAL-015:** All five results are proposed PLAT change/deployment sections.
  They retain draft metadata and proposed wording but do not supply the labelled
  GCP scope/limitation evidence. A plausible deployment result is not an approved
  detailed GCP procedure. Zero evidence coverage is a retrieval miss, not an
  observed successful abstention.

C scores measure conditional-rule evidence and D scores measure corpus-limit
evidence. Their expected behaviors remain metadata; `answer_behavior_evaluated`
is null. The retriever itself neither asks clarification questions nor abstains.

### A.6 Three Successful Retrieval Examples

1. **EVAL-001 — planned versus urgent transfer notification.** Rank 1 is
   `HR-001 v1.0 §4.2 (approved)` at cosine distance 0.1360. The actual section
   preserves both planned and urgent cases. All three precise HR questions have
   complete evidence at rank 1 under approved-only filtering.
2. **EVAL-005 — regenerated Terraform plan.** Top five include
   `PLAT-001 v0.1 §§6.1, 4.3 (proposed)` and
   `COMP-001 v0.1 §8.2 (proposed)`. The rank-1 §6.1 excerpt preserves renewed
   authorized technical review and the conditions for staying within Change
   Management scope/risk/window. Its first distance is 0.1772. The COMP sibling
   adds the sanitized-evidence restriction against ordinary audit packages
   containing raw secret-bearing plans/state. Proxy coverage rises 2/3 → 3/3.
3. **EVAL-013 — unavailable retention duration.** Rank-1
   `COMP-001 v0.1 §11.1 (proposed)` at distance 0.2597 supplies the explicit
   absence of a numeric approved duration. Retrieval successfully finds the
   limitation evidence while preserving draft authority.

### A.7 Three Weakest Cases

1. **EVAL-015 — approved GCP deployment procedure:** 0/3 in both views. Top
   results focus on production CR requirements and execution, rather than the
   requested procedure's absence. General topic relevance does not establish
   scope or approval. Siblings restore the same sections, not the missing units.
2. **EVAL-004 — developer transfer to SRE:** 1/5 in both views. IT-001 §6.3 at
   rank 1 explains that team membership does not grant production administrator
   access, but the top five concentrate on IT role changes. HR removal/exception
   rules and Security approval/MFA evidence are missing. An answer needs more
   than the locally relevant IT handoff paragraph.
3. **EVAL-010 — failing Jenkins job:** 0/3 → 1/3 under the strict proxy. Useful
   hosting/component conditions are present directly, but required Security and
   incident-scope sections are absent. Expansion restores the whole SRE table
   without resolving the missing cross-document evidence.

### A.8 Root-Cause Hypotheses and Scoring Tradeoffs

These are hypotheses from inspected results, not causal conclusions from ablations:

- Dense similarity favors the immediate topic (Jenkins, transfer, deployment).
  A single compound question may not rank all its authority, lifecycle, and
  security dependencies within five slots.
- Several fragments from one section can occupy multiple ranks. This preserves
  useful text but leaves fewer opportunities for another required document.
- Required-sibling expansion repairs fragmentation within a canonical chunk
  family. It is not a dependency-graph search across cited policies.
- Unsupported questions can retrieve convincing general procedures instead of
  explicit limits. Similarity or draft labels alone do not solve this distinction.
- Strict complete-section coverage can under-credit semantic sufficiency. For
  EVAL-005, the rank-1 §6.1 semantic bundle already contains the critical drift
  conditions, though the whole section is complete only by rank 3. For EVAL-006
  and EVAL-010, the directly returned Jenkins table row is relevant; restoring
  unrelated table rows improves the proxy without necessarily adding needed
  Jenkins facts. Do not interpret every proxy increase as an equal semantic gain.

Exact ID overlap would be simpler but could reward a fragment missing conditions.
Complete-source coverage is reproducible and conservative but can over-demand
irrelevant material. Human adjudication against expected claims/qualifiers is the
next step for true M1.8 semantic Recall. No LLM judge or automatic paraphrase-based
grader is introduced here.

## Appendix B — Historical Chroma-Pinecone comparison

This earlier SDK-based experiment used one alternating pass and shared query
embeddings on the same frozen corpus. It does not validate current HTTPX search
or restore an incomplete publication. All usage below is historical reported
read usage, not confirmed billing or a current account balance.

### B.2 Scoring definitions

The evaluator imports M2.5 `score_question` and `summarize` unchanged: `complete-source-section-v1`. A required unit is a document-qualified section. Coverage requires its complete canonical source spans; a parent requires its own text and all dot-delimited descendants. Merely retrieving a section label or fragment is insufficient. This conservative automated proxy can under-credit semantically sufficient excerpts.

- **Recall@5:** fraction of required evidence units completed by the five ranked chunks; reported as a macro-average over questions.
- **Hit@5:** at least one complete required unit is present.
- **Full Evidence Coverage@5:** every required unit is present.
- **MRR@5:** reciprocal of the earliest direct rank whose prefix completes any required unit, averaged across questions.
- **Expanded metrics:** add required siblings to each direct prefix. Siblings have no independent rank or score; expanded MRR is a trigger-prefix metric.

Categories A/B measure evidence for answerable questions; C measures conditional-rule coverage for ambiguous questions; D measures evidence of corpus limits for unsupported questions. The retriever was not credited with clarifying, abstaining or generating an answer. Neither cosine similarity nor these metrics measure answer confidence or faithfulness.

### B.3 Overall results

| Backend / view | Recall@5 | Hit@5 | Full coverage@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Chroma / direct | 55.22% | 86.67% | 26.67% | 0.6944 |
| Chroma / expanded | 64.67% | 93.33% | 40.00% | 0.8667 |
| Pinecone / direct | 55.22% | 86.67% | 26.67% | 0.6911 |
| Pinecone / expanded | 64.67% | 93.33% | 40.00% | 0.8333 |

Sibling expansion increased recall by **9.44 percentage points** on both stores; complete coverage increased from **4/15 to 6/15**. It repaired fragmentation within existing families, without discovering additional cross-policy dependencies. All fifteen per-question missing-evidence sets matched between backends. No authority or provenance violations occurred.

For the nine answerable A/B questions alone, both stores achieved **67.96% direct / 80.00% expanded recall**, **100% Hit@5**, and **33.33% direct / 55.56% expanded full coverage**. Chroma direct MRR was 0.8426 versus Pinecone 0.8370; expanded MRR was 1.0000 for both.

### B.4 Category comparison

Recall, Hit and full coverage matched within every category. Values below show direct → expanded.

| Category | Questions | Recall@5, both | Hit@5, both | Full coverage, both | Chroma MRR | Pinecone MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 3 | 100.00% → 100.00% | 100.00% → 100.00% | 100.00% → 100.00% | 1.0000 → 1.0000 | 1.0000 → 1.0000 |
| B | 6 | 51.94% → 70.00% | 100.00% → 100.00% | 0.00% → 33.33% | 0.7639 → 1.0000 | 0.7556 → 1.0000 |
| C | 3 | 22.22% → 33.33% | 66.67% → 100.00% | 0.00% → 0.00% | 0.4444 → 0.8333 | 0.4444 → 0.6667 |
| D | 3 | 50.00% → 50.00% | 66.67% → 66.67% | 33.33% → 33.33% | 0.5000 → 0.5000 | 0.5000 → 0.5000 |

### B.5 Ranking differences and supported observations

All **75 ranked slots** had matching per-question ID sets; **13/15** questions also had identical order. Chroma order and aggregate metrics reproduced M2.5. The two changes were:

| Question | Record | Chroma rank / similarity | Pinecone rank / similarity |
| --- | --- | ---: | ---: |
| EVAL-006: hybrid Jenkins investigation | SEC-001 §7.2 | 4 / 0.61477804 | 5 / 0.61451350 |
| EVAL-006 | SRE-001 §2 fragment | 5 / 0.61455077 | 4 / 0.61472714 |
| EVAL-012: Rapid Response extension | PLAT-001 §11.3 fragment | 1 / 0.45762408 | 2 / 0.45729116 |
| EVAL-012 | SRE-001 §7.1 | 2 / 0.45740253 | 1 / 0.45737585 |

EVAL-006 direct reciprocal rank changed **1/4 → 1/5**. EVAL-012 expanded reciprocal rank changed **1 → 1/2**, because the fragment triggering complete PLAT §11.3 sibling coverage moved from rank one to rank two. Neither change affected final top-five coverage.

The adapters both sort by distance and then chunk ID. These were close but **not exact ties**; different returned similarity ordering explains the observed rank swaps, rather than a demonstrated difference in the explicit tie-break rule. Eligible ranked sets and expanded context sets matched, giving no evidence that filtering or sibling assembly caused these swaps.

The maximum cross-backend score difference was **0.000531324**. As a local diagnostic, cosine recomputed from the frozen document vectors and shared query vectors differed from Chroma scores by at most **2.85×10⁻⁷**, and Pinecone scores by **0.000531300**. This supports an internal numerical/scoring-implementation difference as a hypothesis. It does **not** establish Pinecone quantization, an ANN algorithm, or any particular service-side mechanism. No ablation or server-internals investigation was performed.

### B.6 Three representative examples

1. **EVAL-001 — HR transfer notification:** both rank approved `HR-001 v1.0 §4.2` first and cover 1/1 required unit. All approved-only results are eligible HR records.
2. **EVAL-005 — regenerated Terraform plan:** both retrieve proposed PLAT-001 §6.1, §4.3 and COMP-001 §8.2. Direct coverage is 2/3; sibling expansion completes 3/3. This retains technical approval and Change Management conditions without treating a fragment as an independent complete rule.
3. **EVAL-013 — approved retention duration:** both rank proposed COMP-001 §11.1 first and retrieve the required corpus-limit unit. That is evidence about a missing approved duration, not a numeric duration or proof of successful abstention.

### B.7 Weakest cases and remaining gaps

- **EVAL-015 — approved GCP deployment procedure:** 0/3 required units in either view. Results concern general production changes, not the required scope/authority limits: PLAT-001 §1.1, §2 and SEC-001 §6.1 remain missing.
- **EVAL-004 — transfer into SRE with administrator access:** 1/5 in both views. IT-001 §6.3 is found, but HR-001 §4.3/§4.4 and SEC-001 §5.2/§4.1 are missing. Topic relevance does not supply all lifecycle and approval obligations.
- **EVAL-010 — ambiguous Jenkins failure:** 0/3 direct and 1/3 expanded. Restoring SRE-001 §6.1 does not recover SEC-001 §7.2 or SRE-001 §1.1. Hosting/component conditions remain important; the retriever itself asks no clarification.

Other gaps include IT-001 §5.2 for workload-secret access, SRE incident scope/declaration, Rapid Response amendment boundaries, and SEC workload-credential rules. Fragment competition and compound cross-policy questions are plausible limitations of dense top-five retrieval; this experiment did not test remedies or prove a unique cause.

### B.8 Latency

| Measurement | Mean | Median | Maximum / first query | Remaining-query mean |
| --- | ---: | ---: | ---: | ---: |
| Query Encoding | 241.68 ms | 12.35 ms | 3448.15 ms | 12.64 ms |
| Chroma | 21.85 ms | 20.62 ms | 40.99 ms | 20.48 ms |
| Pinecone | 301.66 ms | 269.70 ms | 549.80 ms | 283.94 ms |

Cached model load took **10.564 s**; total setup, readiness and query preparation took **18.898 s**. Encoding happened before either backend search. First encoding was **3.448 s**, with later encodes averaging **12.64 ms**. First retrieval was **41.00 ms Chroma / 549.80 ms Pinecone**. Connections and clients were already initialized by status checks, so these are first-query observations, not controlled cold-machine or cold-service measurements.

Retrieval timing includes adapter readiness/source/target checks, search, and selected-record/sibling validation. It excludes model encoding, scoring and artifact IO. Chroma performs local count/payload checks; Pinecone includes network requests and an index-description check per question. These are end-to-end adapter timings, not isolated database-engine latency. One alternating pass on an Intel Mac cannot establish throughput, tail percentiles or production latency guarantees. M2.5 timings included encoding/full audits and are not directly comparable to these timings.

### B.9 Pinecone usage

- **58 logical HTTP calls total:** 20 descriptions, 15 searches and 23 required-sibling fetch requests.
- The descriptions comprise existing-index and namespace status checks plus per-query target checks. Namespace counts were checked before and after the run.
- Query phase: **53 calls** (15 target descriptions, 15 searches, 23 sibling fetches).
- **38 reported read units**, all from the query phase. No full-index fetch, record listing, visibility probes, upsert, deletion or resource mutation occurred.
- Logical counts do not expose SDK-internal retries. No write-unit usage was reported; no monetary cost is estimated, and credit/billing state was not inspected.

## Appendix C — Earlier hybrid timing experiment

These values belong to the preceding hybrid experiment, separate from the
three-pass measurements in section 5. Their differing values are not revisions
to that later run.

### Context safeguards and latency

Generation readiness is checked offline using the existing full prompt builder: 32,768 context tokens, 1,024 output reservation and 512 margin. Counts use the existing conservative Qwen byte bound, not billed tokens or MiniLM's 256-token embedding limit.

| Strategy | Mean expanded chunks | Mean expanded characters | Mean / maximum context bound | Full-prompt-ready |
| --- | ---: | ---: | ---: | ---: |
| Dense | 7.40 | 6,761 | 13,369 / 26,245 | 14/15 |
| BM25 | 7.67 | 7,263 | 13,922 / 28,701 | 14/15 |
| Hybrid | 7.87 | 7,435 | 14,160 / 26,592 | 13/15 |

EVAL-006 exceeds the complete prompt budget in all arms. EVAL-010 also exceeds it in hybrid. The evidence-only packages fit, but instructions, schema and question overhead make the full request too large. The existing generator rejects those requests; no mandatory context is silently trimmed. Treating failed preparation as no evidence delivered gives aggregate delivered recall of **58.00% dense, 60.11% BM25, 61.56% hybrid**, separate from the frozen retrieval metrics. Readiness itself says nothing about semantic sufficiency.

Final measured warm retrieval times (45 observations per arm, excluding query encoding and context construction):

| Strategy | Mean | Median | Maximum |
| --- | ---: | ---: | ---: |
| Dense | 19.02 ms | 18.54 ms | 26.22 ms |
| BM25 | 7.67 ms | 7.80 ms | 10.95 ms |
| Hybrid | 40.31 ms | 40.35 ms | 50.89 ms |

These include source/readiness checks, selected payload verification and sibling work. Dense fetches five direct hits, while hybrid fetches twenty dense candidates and their siblings before fusion; this is whole-adapter cost, not isolated search-engine timing. First correctness-pass averages were 19.64, 7.68 and 43.69 ms respectively. Old M2.5 timing included embedding and different audit work and is not directly comparable.

Final model loading took 10.72 s; setup including verification, readiness, model load and all query encoding took 16.95 s. Lexical scorer construction alone took 8.44 ms. Query encoding across all 16 cases averaged 204.39 ms, median 11.82 ms, maximum 3.094 s. These observations are not a controlled cold-machine benchmark. Catalog verification remains bundled with setup; it was not separately timed, so lexical build time must not be presented as complete BM25 startup cost. Complete lexical startup time was not measured separately.

## Appendix D — Exact frozen questions

Question text is copied from the unchanged public dataset, not generated answers.
The expected behavior is a benchmark label, not an observed model outcome.

| ID | Mode | Expected behavior | Question |
| --- | --- | --- | --- |
| EVAL-001 | approved_only | answer | When must HR notify IT about a planned internal transfer, and what changes for an urgent transfer without advance notice? |
| EVAL-002 | approved_only | answer | By when must access from an employee's previous role be removed after a transfer, and what is the exception boundary? |
| EVAL-003 | approved_only | answer | When must IT disable a departing employee's access, including an urgent involuntary termination? |
| EVAL-004 | draft_review | answer | A developer is transferring to SRE and the receiving manager supports production administrator access. Does the transfer authorize that access, what decisions are needed, and what happens to old access while the new request is pending? |
| EVAL-005 | draft_review | answer | A production Terraform plan is regenerated after drift is detected. When is renewed Platform review enough without another Change Management approval, when must work stop for amendment or escalation, and what evidence should be retained? |
| EVAL-006 | draft_review | answer | Who investigates failures in cloud-hosted versus on-premises Jenkins, and how do application teams and SRE participate when a pipeline crosses both environments and production service is affected? |
| EVAL-007 | draft_review | answer | An AKS workload uses a managed identity to retrieve a Key Vault secret. Does that identity or the team's vault-administrator access authorize secret retrieval, and what access decisions and integration checks are required? |
| EVAL-008 | draft_review | answer | A production deployment fails and users are still affected. Who assesses and declares a reliability incident, and can it be closed once a later Jenkins job turns green? |
| EVAL-009 | draft_review | answer | A Venafi certificate renewal failure disrupts a cloud-hosted production service. What belongs to Platform, Security, and SRE, and does the corpus identify the certificate-service operator? |
| EVAL-010 | draft_review | clarify | The Jenkins job is failing. Which team should fix it? |
| EVAL-011 | draft_review | clarify | My manager approved 'production access'. Is that enough to provision it? |
| EVAL-012 | draft_review | clarify | Rapid Response says we need more time. Can we keep deploying? |
| EVAL-013 | draft_review | abstain | Exactly how many years is the approved retention period for production change and access-review evidence? |
| EVAL-014 | draft_review | abstain | Which exact MFA method is approved for administrators, and what fixed credential-rotation interval must our workloads use? |
| EVAL-015 | draft_review | abstain | Give me the detailed approved GCP production deployment procedure, including the exact execution steps and approvals. |
