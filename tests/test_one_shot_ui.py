"""Local one-shot UI: synthetic authorities, real QA validation, fake providers."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import test_authority_handoff as fixtures
from test_context import chunk, hit
from test_qa import FakeLLM, FixtureStore, authored_claims
from northstar_atlas import authority_handoff as handoff, local_live
from northstar_atlas.generation import prepare_answer
from northstar_atlas.ingestion.document_loader import APPROVED
from northstar_atlas.live_budget import BudgetError
from northstar_atlas.qa import QAService
from northstar_atlas.ui.runtime import ResourceOwner, Submission, POLICY_SEARCH


class OneShotUITests(unittest.TestCase):
    def setUp(self):
        original_main = sys.modules['__main__']
        self.addCleanup(sys.modules.__setitem__, '__main__', original_main)
        f = fixtures.HandoffTests('runTest'); f.setUp(); self.addCleanup(f.doCleanups)
        self.f = f; self.preview, _ = f.ready()
        self.root, self.budget, self.spec = f.source.root, f.destination, f.spec
        os.environ['NORTHSTAR_AUTHORITY_HANDOFF_PLAN'] = self.preview.plan_sha256
        self.enterContext(patch.object(Path, 'cwd', return_value=self.root))
        self.enterContext(patch('northstar_atlas.live_budget.configured_budget', return_value=self.budget))
        self.enterContext(patch.object(local_live, 'configured_budget', return_value=self.budget))
        self.opens = []; self.closed = []
        source = chunk('Only within the approved scope.\n', doc='HR-001', status=APPROVED,
                       effective=self.spec.reference_date)
        self.store = FixtureStore([hit(source)])
        prepared = prepare_answer(self.spec.question, self.store.hits,
                                  mode=self.spec.mode, reference_date=self.spec.reference_date)
        self.provider = FakeLLM(authored_claims(prepared))
        self.provider.requires_durable_budget = True
        self.enterContext(patch('northstar_atlas.ui.runtime.open_qa_service', side_effect=self.open_service))
        self.enterContext(patch('northstar_atlas.nebius.load_settings', side_effect=AssertionError('credentials forbidden')))
        self.enterContext(patch('northstar_atlas.nebius.NebiusProvider', side_effect=AssertionError('real provider forbidden')))
        self.enterContext(patch('northstar_atlas.qa.open_retriever', side_effect=AssertionError('real retrieval forbidden')))
        self.enterContext(patch.object(handoff, 'prepare_handoff', side_effect=AssertionError('UI must not prepare handoff')))
        self.enterContext(patch.object(handoff, 'retire_source', side_effect=AssertionError('UI must not retire authority')))
        self.enterContext(patch.object(handoff, 'activate_destination', side_effect=AssertionError('UI must not activate authority')))
        st.cache_resource.clear(); self.addCleanup(st.cache_resource.clear)

    @contextmanager
    def open_service(self, root, **options):
        self.assertEqual(root, self.root)
        flow = json.loads(self.budget.path.read_bytes())['workflows'][-1]
        self.assertEqual((flow['id'], flow['status'], flow['calls']), (self.spec.invocation_id, 'reserved', []))
        self.assertEqual(options['backend'], 'pinecone')
        options['reservation'].check_coordinator(self.budget)
        self.opens.append(True)
        try:
            yield QAService(self.store, object(), self.provider, corpus_fingerprint='fixture-corpus',
                            backend='pinecone', budget=self.budget)
        finally:
            self.closed.append(True)

    def owner(self):
        owner = ResourceOwner(self.root, live_enabled=True, pinecone_enabled=True, budget=self.budget)
        self.addCleanup(owner.close)
        return owner

    def submission(self, **changes):
        return replace(Submission('ui-turn', self.spec.question, self.spec.mode, 'pinecone',
                                  self.spec.reference_date), **changes)

    def app(self):
        app = AppTest.from_file(str(self.root/'src/northstar_atlas/ui/app.py')).run(timeout=20)
        self.assertFalse(app.exception)
        return app

    def live_app(self):
        app = self.app(); app.radio(key='experience').set_value('Live Q&A').run()
        self.assertFalse(app.exception)
        return app

    def test_authorized_ui_binds_fixed_id_and_records_after_settlement(self):
        app = self.live_app()
        self.assertNotIn('workflow', [x.key for x in app.selectbox])
        self.assertEqual(app.selectbox(key='backend').value, 'pinecone')
        self.assertTrue(app.selectbox(key='backend').disabled)
        self.assertEqual(app.selectbox(key='mode').value, self.spec.mode)
        self.assertTrue(app.selectbox(key='mode').disabled)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(any('One approved question' in x.value for x in app.caption))
        app.checkbox(key='paid_opt_in').check().run()
        self.assertFalse(app.chat_input[0].disabled)
        from northstar_atlas.ui import runtime
        project = runtime.project_result
        def displayed(result, mode):
            value = json.loads(self.budget.path.read_bytes())
            self.assertEqual(value['workflows'][-1]['status'], 'closed')
            record = self.root/f'.local/linear-demo-results/{self.spec.invocation_id}.json'
            self.assertEqual(json.loads(record.read_bytes())['status'], 'accepted')
            return project(result, mode)
        with patch.object(runtime, 'project_result', side_effect=displayed):
            app.chat_input[0].set_value(self.spec.question).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['conversation'].history[-1].view.kind, 'answer')
        self.assertTrue(app.session_state['conversation'].history[-1].view.citations)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(any('already been attempted' in x.value for x in app.warning))
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(len(self.closed), 1)
        self.assertEqual(self.store.calls[0][2]['reference_date'], self.spec.reference_date)

    def test_wrong_question_does_not_claim_approved_id_or_open_resources(self):
        owner = self.owner()
        result = owner.submit(self.submission(question='A different question?'), consent=True)
        self.assertEqual(result.code, 'one_shot_mismatch')
        self.assertEqual(self.opens, [])
        self.assertEqual(len(json.loads(self.budget.path.read_bytes())['workflows']), 7)
        self.assertFalse((self.root/f'.local/linear-demo-results/{self.spec.invocation_id}.json').exists())
        self.assertEqual(owner.submit(self.submission(turn_id='correct'), consent=True).kind, 'answer')

    def test_duplicate_sessions_refresh_and_restarted_owner_cannot_reuse_allowance(self):
        first, second = self.live_app(), self.live_app()
        first.checkbox(key='paid_opt_in').check().run()
        self.assertFalse(second.checkbox(key='paid_opt_in').value)
        self.assertTrue(second.chat_input[0].disabled)
        first.chat_input[0].set_value(self.spec.question).run(timeout=20)
        first.run(); second.run()
        self.assertTrue(first.chat_input[0].disabled)
        self.assertTrue(second.chat_input[0].disabled)
        first.button(key='clear_history').click().run()
        self.assertTrue(first.chat_input[0].disabled)
        st.cache_resource.clear()
        restarted = self.live_app()
        self.assertTrue(restarted.chat_input[0].disabled)
        self.assertFalse(restarted.checkbox(key='paid_opt_in').value)
        self.assertEqual(self.owner().submit(self.submission(turn_id='fresh-process-id'), consent=True).code, 'one_shot_spent')
        with self.assertRaises(BudgetError):
            local_live.check_budget(self.root, enable_graph=False, backend='pinecone')
        self.assertEqual(len(self.provider.calls), 1)

    def test_invalid_output_withheld_and_never_regenerated(self):
        self.provider.body = '{invalid structured response'
        app = self.live_app(); app.checkbox(key='paid_opt_in').check().run()
        app.chat_input[0].set_value(self.spec.question).run(timeout=20)
        view = app.session_state['conversation'].history[-1].view
        self.assertEqual(view.code, 'response_rejected')
        self.assertFalse(any('{invalid' in x.value for x in app.text))
        self.assertEqual(len(self.provider.calls), 1)
        app.run(); self.assertTrue(app.chat_input[0].disabled)

    def test_missing_expired_or_conflicting_configuration_has_no_fallback(self):
        changes = [dict(NORTHSTAR_AUTHORITY_HANDOFF_PLAN=''),
                   dict(NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL='wrong'),
                   dict(NORTHSTAR_DEMO_BATCH_ENABLED='1'), dict(NORTHSTAR_UI_ENABLE_GRAPH='1')]
        for env in changes:
            with self.subTest(env=env), patch.dict(os.environ, env):
                app = self.live_app()
                self.assertTrue(app.chat_input[0].disabled)
                self.assertEqual(self.owner().submit(self.submission(), consent=True).code, 'budget_unavailable')
        self.f.f.now += timedelta(hours=2)
        self.assertEqual(self.owner().submit(self.submission(), consent=True).code, 'budget_unavailable')
        self.f.f.now -= timedelta(hours=2)
        self.budget.path.unlink()
        self.assertEqual(self.owner().submit(self.submission(), consent=True).code, 'budget_unavailable')
        self.assertEqual(self.opens, []); self.assertEqual(self.provider.calls, [])

    def test_bound_fields_and_session_consent_checked_before_resources(self):
        owner = self.owner()
        self.assertEqual(owner.submit(self.submission(), consent=False).code, 'opt_in_required')
        for change in (dict(mode='draft_review'), dict(backend='chroma'),
                       dict(reference_date=self.spec.reference_date+timedelta(days=1)),
                       dict(workflow='langgraph_semantic')):
            with self.subTest(change=change):
                self.assertEqual(owner.submit(self.submission(**change), consent=True).kind, 'error')
        self.assertEqual(self.opens, [])
        self.assertEqual(len(json.loads(self.budget.path.read_bytes())['workflows']), 7)

    def test_offline_demo_and_policy_search_do_not_inspect_authority(self):
        with patch.object(handoff, 'inspect_invocation', side_effect=AssertionError('authority read forbidden')):
            app = self.app(); self.assertEqual(app.radio(key='experience').value, 'Offline demo')
            app.button(key='run_demo').click().run()
            self.assertTrue(any('authored examples' in x.value for x in app.info))
            app.radio(key='experience').set_value(POLICY_SEARCH).run()
            self.assertFalse(app.chat_input[0].disabled)
            self.assertTrue(app.checkbox(key='paid_opt_in').disabled)
            self.assertEqual(app.selectbox(key='backend').value, 'chroma')
            self.assertFalse(app.exception)
        self.assertEqual(self.opens, [])

    def test_startup_checks_unused_successor_without_mutating_or_activating(self):
        before = self.budget.path.read_bytes()
        budget = local_live.check_budget(self.root, enable_graph=False, backend='pinecone')
        self.assertIs(budget, self.budget)
        self.assertEqual(self.budget.path.read_bytes(), before)
        for opts in (dict(enable_graph=True,backend='pinecone'), dict(enable_graph=False,backend='chroma')):
            with self.assertRaises(BudgetError): local_live.check_budget(self.root, **opts)
        self.assertEqual(self.opens, [])

    def test_timeout_and_settlement_failure_never_reopen_the_ui_allowance(self):
        self.provider.error = TimeoutError('synthetic timeout, no transport')
        owner = self.owner()
        self.assertEqual(owner.submit(self.submission(), consent=True).code, 'provider_failed')
        self.assertEqual(json.loads(self.budget.path.read_bytes())['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(self.owner().submit(self.submission(turn_id='retry'), consent=True).code, 'one_shot_spent')
        self.assertEqual(len(self.provider.calls), 1)

    def test_failed_durable_settlement_withholds_accepted_content(self):
        owner = self.owner()
        settle = self.budget._settle
        def fail(value, **kw):
            with patch.object(self.budget, '_write', side_effect=OSError('synthetic persistence failure')):
                return settle(value, **kw)
        with patch.object(self.budget, '_settle', side_effect=fail):
            result = owner.submit(self.submission(), consent=True)
        self.assertEqual(result.code, 'budget_unavailable')
        self.assertTrue(self.budget.pending_path.exists())
        record = json.loads((self.root/f'.local/linear-demo-results/{self.spec.invocation_id}.json').read_bytes())
        self.assertEqual(record['status'], 'outcome_unknown')
        self.assertNotIn('answer_text', record)
        self.assertEqual(self.owner().submit(self.submission(turn_id='retry'), consent=True).code, 'budget_unavailable')
        self.assertEqual(len(self.provider.calls), 1)


if __name__ == '__main__':
    unittest.main()
