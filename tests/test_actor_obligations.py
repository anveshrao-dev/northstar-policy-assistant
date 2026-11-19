"""Reviewed actor attribution against genuine synthetic policies; no provider IO."""
import json
from dataclasses import replace
from decimal import Decimal
import socket
import unittest
from unittest.mock import patch

from test_context import AS_OF, ROOT, hit
from test_qa import FakeLLM, FixtureStore
from northstar_atlas import generation as g
from northstar_atlas.evidence import format_citation
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, load_documents
from northstar_atlas.qa import QAService, QAFailure, QASuccess

QUESTION = "What steps must HR and the employee's manager complete before an internal role transfer takes effect?"
INITIATION = ('A role change must be approved by the receiving manager and recorded by HR '
              'before the transfer effective date.')


class ActorObligationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')))
        documents = load_documents(ROOT, reference_date=AS_OF)
        self.hr = next(d for d in documents if d.document_id == 'HR-001')
        self.chunks = {c.section_id: c for c in chunk_document(self.hr).chunks}
        self.hits = [hit(self.chunks[s]) for s in ('4.1', '3.2', '4.2', '4.3')]
        self.prepared = g.prepare_answer(QUESTION, self.hits, reference_date=AS_OF)

    def body(self, text, section='4.1', quote=INITIATION):
        group = next(g for g in self.prepared.context.groups if g.citation.section_id == section)
        return dict(kind='answer', main_answer=dict(obligations={}, claims=[dict(
            text=text, support=[dict(citation=format_citation(group.citation), quote=quote)])]),
            missing_information=[], clarification_question=None, reason='supported')

    def test_hr_approval_substitution_with_valid_citation_and_quote_is_withheld(self):
        provider = FakeLLM(self.body('HR must approve the role change.'))
        service = QAService(FixtureStore(self.hits), object(), provider,
                            corpus_fingerprint='fixture', backend='chroma')
        result = service.ask(QUESTION, reference_date=AS_OF)
        self.assertIsInstance(result, QAFailure)
        self.assertEqual(result.public.code, 'response_rejected')
        self.assertEqual(result.diagnostics.provider_calls, 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn('HR must approve', result.public.message)

    def validate(self, body, context=None, **kwargs):
        return g.validate_answer(json.dumps(body), context or self.prepared.context, **kwargs)

    def test_correct_reviewed_multirole_sentence_and_exact_source_are_unchanged(self):
        for text in (INITIATION, self.chunks['4.1'].text.strip(),
                     'The receiving manager must approve the role change and HR must record it before the transfer effective date.',
                     'Before the transfer effective date, the receiving manager must approve the role change and HR must record it.'):
            with self.subTest(text=text):
                answer = self.validate(self.body(text))
                self.assertEqual(answer.claims[0].text, text)
                self.assertEqual(answer.coverage.status, 'not_assessed')

    def test_correct_four_role_explanation_preserves_other_sections(self):
        body = self.body(INITIATION)
        for section in ('3.2', '4.2', '4.3'):
            group = next(g for g in self.prepared.context.groups if g.citation.section_id == section)
            quote = group.spans[0].text.strip()
            body['main_answer']['claims'].extend(self.body(quote, section, quote)['main_answer']['claims'])
        answer = self.validate(body)
        self.assertEqual(len(answer.claims), 4)
        self.assertEqual({s.citation.section_id for c in answer.claims for s in c.support}, {'4.1','3.2','4.2','4.3'})

    def test_substitutions_ambiguity_negation_and_appended_false_claim_are_rejected(self):
        cases = ('HR approves the role change and the receiving manager records it.',
                 'The current manager must approve the role change.',
                 'They must approve and record it before the transfer takes effect.',
                 'HR and the receiving manager must approve the role change.',
                 INITIATION + ' HR must approve it too.',
                 'It is not true that ' + INITIATION,
                 'HR may approve the role change.',
                 'Approval can be skipped before transfer.')
        for text in cases:
            with self.subTest(text=text), self.assertRaises(g.GenerationError):
                self.validate(self.body(text))

    def test_missing_reviewed_source_is_withheld_even_with_a_valid_other_quote(self):
        group = next(g for g in self.prepared.context.groups if g.citation.section_id == '4.1')
        span = group.spans[0]
        # A structurally coherent partial supplied span contains only HR's record fields.
        start = span.text.index('HR records')
        partial = replace(span, text=span.text[start:], source_start=span.source_start+start)
        context = replace(self.prepared.context, groups=(replace(group, spans=(partial,)),))
        body = self.body('HR must approve the role change.', quote=partial.text.strip())
        with self.assertRaises(g.GenerationError): self.validate(body, context)

    def test_missing_citation_still_fails_existing_membership_check(self):
        context = replace(self.prepared.context, groups=tuple(
            g for g in self.prepared.context.groups if g.citation.section_id != '4.1'))
        with self.assertRaises(g.GenerationError): self.validate(self.body(INITIATION), context)

    def test_ambiguous_duplicate_source_span_is_withheld(self):
        group = next(g for g in self.prepared.context.groups if g.citation.section_id == '4.1')
        context = replace(self.prepared.context, groups=(replace(group, spans=group.spans*2),))
        with self.assertRaises(g.GenerationError): self.validate(self.body(INITIATION), context)

    def test_correct_clause_cannot_launder_an_unrelated_assertion(self):
        with self.assertRaises(g.GenerationError):
            self.validate(self.body(INITIATION + ' All transfers automatically grant production access.'))

    def test_other_section_is_not_subject_to_initiation_vocabulary(self):
        text = 'HR must notify IT Operations at least two business days before a planned internal transfer.'
        quote = 'HR must notify IT Operations at least **two business days before** a planned internal transfer.'
        answer = self.validate(self.body(text, '4.2', quote))
        self.assertEqual(answer.claims[0].text, text)

    def test_different_policy_can_assign_approval_to_hr(self):
        from test_context import chunk
        source = chunk('HR must approve the training budget.', doc='TRAIN-001', status=APPROVED, effective=AS_OF)
        prepared = g.prepare_answer('Who approves training?', [hit(source)], reference_date=AS_OF)
        citation = prepared.context.groups[0].citation
        body = self.body(source.text)
        body['main_answer']['claims'][0]['support'] = [dict(citation=format_citation(citation), quote=source.text)]
        self.assertEqual(self.validate(body, prepared.context).claims[0].text, source.text)

    def test_draft_snapshot_never_inherits_approved_actor_rule(self):
        from test_context import chunk
        source = chunk('HR must approve the role change.', doc='HR-001', section='4.1', status=DRAFT)
        prepared = g.prepare_answer('Who approves?', [hit(source)], mode='draft_review', reference_date=AS_OF)
        citation = prepared.context.groups[0].citation
        body = self.body(source.text)
        body['main_answer']['claims'][0]['support'] = [dict(citation=format_citation(citation), quote=source.text)]
        answer = self.validate(body, prepared.context)
        self.assertEqual(answer.claims[0].support[0].citation.status, DRAFT)
        self.assertIn('(proposed)', g.render_answer(answer))
        filtered = g.prepare_answer('Who approves?', [hit(source)], reference_date=AS_OF)
        with self.assertRaises(g.GenerationError): self.validate(body, filtered.context)

    def test_approved_snapshot_in_draft_review_still_enforces_reviewed_roles(self):
        context = g.prepare_answer(QUESTION, self.hits, mode='draft_review', reference_date=AS_OF).context
        with self.assertRaises(g.GenerationError): self.validate(self.body('HR must approve the role change.'), context)

    def test_evidence_linked_obligation_cannot_bypass_actor_guard(self):
        from northstar_atlas.grounding import EvidenceObligation
        citation = self.prepared.context.groups[0].citation
        obligation = EvidenceObligation('initiation', citation, INITIATION, 'evidence_linked')
        body = self.body('HR must approve the role change.')
        claim = body['main_answer']['claims'].pop()
        body['main_answer']['obligations'] = dict(initiation=dict(
            statement=claim['text'], support=claim['support'], unresolved_reason=None))
        with self.assertRaises(g.GenerationError): self.validate(body, obligations=(obligation,))
        body['main_answer']['obligations']['initiation']['statement'] = INITIATION
        self.assertEqual(self.validate(body, obligations=(obligation,)).kind, 'answer')

    def test_no_evidence_abstention_does_not_invent_actor_roles_or_call_provider(self):
        provider = FakeLLM(error=AssertionError('no provider expected'))
        service = QAService(FixtureStore(), object(), provider, corpus_fingerprint='fixture', backend='chroma')
        result = service.ask(QUESTION, reference_date=AS_OF)
        self.assertIsInstance(result, QASuccess)
        self.assertEqual(result.public.kind, 'abstain')
        self.assertEqual(provider.calls, [])

    def test_rejection_preserves_durable_attempt_cost_and_duplicate_protection(self):
        from budget_fixture import controlled_budget
        from checkout_fixture import checkout
        from northstar_atlas.live_budget import WorkflowRequest, LINEAR, BudgetError
        from northstar_atlas.result_capture import capture_result
        root = checkout(self)
        budget = controlled_budget(self, root, calls=3)
        provider = FakeLLM(self.body('HR must approve the role change.'))
        service = QAService(FixtureStore(self.hits), object(), provider,
                            corpus_fingerprint='fixture', backend='chroma', budget=budget)
        spec = WorkflowRequest('synthetic-actor-check', LINEAR, QUESTION, 'approved_only', 'chroma', AS_OF)
        def execute():
            with budget.reserve(spec) as reservation:
                result = service.ask(QUESTION, reference_date=AS_OF, reservation=reservation)
            return result
        result = capture_result(execute, root=root, invocation_id=spec.invocation_id)
        self.assertIsInstance(result, QAFailure)
        self.assertEqual(result.public.code, 'response_rejected')
        ledger = json.loads(budget.path.read_bytes())
        flow = ledger['workflows'][0]
        self.assertEqual(flow['status'], 'closed')  # Provider returned; answer rejected locally.
        self.assertEqual(len(flow['calls']), 1)
        self.assertEqual(flow['calls'][0]['status'], 'returned')
        self.assertEqual(budget.inspect_history().reserved_usd, Decimal('0.003687'))
        self.assertEqual(budget.inspect_history().attempts_consumed, 1)
        self.assertFalse(budget.pending_path.exists())
        saved = json.loads((root/'.local/linear-demo-results/synthetic-actor-check.json').read_bytes())
        self.assertEqual(saved['status'], 'not_accepted')
        self.assertNotIn('answer_text', saved)
        with self.assertRaises(BudgetError):
            with budget.reserve(spec): self.fail('duplicate must not execute')
        self.assertEqual(len(provider.calls), 1)

    def test_graph_deterministic_gate_withholds_before_semantic_check(self):
        from test_bm25 import catalog
        from test_orchestration_semantic import PhaseBSemanticGraphTests, ScopedStore
        fixture = PhaseBSemanticGraphTests()
        fixture.setUp()
        fixture.hits = tuple(self.hits)
        fixture.cat = catalog([h.chunk for h in self.hits])
        service, _, _, provider = fixture.make(first=ScopedStore(self.hits),
            body=self.body('HR must approve the role change.'))
        result = service.ask(QUESTION, reference_date=AS_OF)
        self.assertIsInstance(result, QAFailure)
        self.assertEqual(result.public.code, 'response_rejected')
        requests = [json.loads(r.user) for r in provider.calls]
        self.assertEqual(sum('evidence' in r for r in requests), 1)
        self.assertFalse(any('candidate_text' in r for r in requests))


if __name__ == '__main__':
    unittest.main()
