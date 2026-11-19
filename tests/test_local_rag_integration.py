"""Actual QAService, Chroma, pinned cached MiniLM and mocked Nebius HTTP.

Only the credential loader and HTTP transport are fake. The cached model is
shared across disposable checkout/index fixtures; no operational index or
accounting is opened. These are contract tests, not live answer-quality scores.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

import httpx
import streamlit as st
from streamlit.testing.v1 import AppTest

from checkout_fixture import ROOT
import test_local_test_authority as fixtures
from test_generation_v2 import payload
from northstar_atlas import local_test_authority as local, nebius
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT
from northstar_atlas.ui.runtime import ResourceOwner, Submission


@unittest.skipUnless(os.environ.get('NORTHSTAR_RUN_MODEL_TESTS') == '1', 'requires explicitly enabled cached MiniLM checks')
class LocalRAGIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from northstar_atlas.local_embeddings import MiniLMProvider
        from northstar_atlas.vector_store.preparation import prepare_corpus
        cls.embedding = MiniLMProvider(ROOT)
        cls.corpus = prepare_corpus(ROOT, cls.embedding)
        assert len(cls.corpus.records) == 189

    def setUp(self):
        original_main = sys.modules['__main__']
        self.addCleanup(sys.modules.__setitem__, '__main__', original_main)
        f = fixtures.FreshAuthorityTests('runTest'); f.setUp(); self.addCleanup(f.doCleanups)
        self.f = f; self.root = f.root
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network forbidden')))
        self.enterContext(patch.object(Path, 'cwd', return_value=self.root))
        from northstar_atlas.vector_store import ChromaStore
        store = ChromaStore(self.root, self.corpus.config)
        self.assertEqual(store.synchronize(self.corpus.records, corpus_fingerprint=self.corpus.fingerprint).total, 189)
        search = ChromaStore.search
        self.hits = []
        def searched(store, *args, **kw):
            found = search(store, *args, **kw); self.hits.append(found); return found
        self.enterContext(patch.object(ChromaStore, 'search', new=searched))
        self.enterContext(patch('northstar_atlas.local_embeddings.MiniLMProvider', return_value=self.embedding))
        self.sent = []; self.opens = []; self.behavior = 'answer'
        real_provider = nebius.NebiusProvider
        def settings(root):
            self.assertEqual(root, self.root)
            self.assert_reserved(False)
            self.opens.append(True)
            return nebius.NebiusSettings('synthetic-not-a-real-credential')
        self.enterContext(patch.object(nebius, 'load_settings', side_effect=settings))
        self.enterContext(patch.object(nebius, 'NebiusProvider', side_effect=lambda settings, **kw:
            real_provider(settings, transport=httpx.MockTransport(self.transport), **kw)))
        st.cache_resource.clear(); self.addCleanup(st.cache_resource.clear)

    def authorize(self, *, mode='approved_only', question='How should an employee request time off and who approves it?'):
        self.f.spec = replace(self.f.spec, question=question, mode=mode)
        self.f.prepare(); self.f.apply()
        self.budget = self.f.budget
        self.enterContext(patch('northstar_atlas.live_budget.configured_budget', return_value=self.budget))
        return self.f.spec

    def assert_reserved(self, dispatched):
        flow = json.loads(self.f.budget.path.read_bytes())['workflows'][0]
        self.assertEqual(flow['status'], 'reserved')
        self.assertEqual(len(flow['calls']), int(dispatched))
        if dispatched: self.assertEqual(flow['calls'][0]['status'], 'started')

    def transport(self, request):
        self.assert_reserved(True)
        self.sent.append(request.method)
        body = json.loads(request.content)
        self.evidence = json.loads(body['messages'][1]['content'])['evidence']
        if self.behavior == 'timeout': raise httpx.ReadTimeout('synthetic timeout')
        if self.behavior == 'envelope': return httpx.Response(200, json={'choices':[]})
        answer = payload(kind='abstain' if self.behavior == 'abstain' else 'answer')
        groups = self.evidence['approved_evidence'] + self.evidence['proposed_evidence']
        if self.behavior != 'abstain':
            for g in groups[:2]:
                # Exact source sentences make the deterministic fixture support
                # explicit; this does not simulate a model's semantic judgment.
                quote = next(line.strip() for p in g['passages'] for line in p['text'].splitlines()
                             if len(line.strip()) > 40 and not line.startswith('#'))
                answer['main_answer']['claims'].append({'text':quote, 'support':[
                    {'citation':g['citation'], 'quote':quote}]})
            if self.behavior == 'citation': answer['main_answer']['claims'][0]['support'][0]['citation'] = 'invented citation'
            if self.behavior == 'unsupported': answer['main_answer']['claims'][0]['support'] = []
        value = dict(model=nebius.MODEL_ID, choices=[dict(message={'role':'assistant', 'content':json.dumps(answer)},
            finish_reason='stop')], usage=dict(prompt_tokens=100, completion_tokens=100, total_tokens=200))
        return httpx.Response(200, json=value)

    def owner(self):
        owner = ResourceOwner(self.root, live_enabled=True, budget=self.budget)
        self.addCleanup(owner.close)
        return owner

    def submit(self, *, owner=None, **changes):
        spec = self.f.spec
        submission = Submission('synthetic-ui-click', spec.question, spec.mode, 'chroma', spec.reference_date)
        return (owner or self.owner()).submit(replace(submission, **changes), consent=True)

    def record(self):
        return json.loads((self.root/f'.local/linear-demo-results/{self.f.spec.invocation_id}.json').read_bytes())

    def test_actual_streamlit_answer_settles_and_records_before_display_and_cannot_replay(self):
        self.authorize()
        from northstar_atlas.ui import runtime
        present = runtime.project_result
        def checked(result, mode):
            self.assertEqual(json.loads(self.budget.path.read_bytes())['workflows'][0]['status'], 'closed')
            self.assertEqual(self.record()['status'], 'accepted')
            self.assertEqual(self.record()['diagnostics']['token_usage']['total_tokens'], 200)
            return present(result, mode)
        with patch.object(runtime, 'project_result', side_effect=checked):
            app = AppTest.from_file(str(self.root/'src/northstar_atlas/ui/app.py')).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='experience').value, 'Offline demo')
            app.radio(key='experience').set_value('Live Q&A').run()
            self.assertTrue(app.chat_input[0].disabled)
            self.assertEqual(app.selectbox(key='backend').value, 'chroma')
            self.assertTrue(app.selectbox(key='backend').disabled)
            app.checkbox(key='paid_opt_in').check().run()
            app.chat_input[0].set_value(self.f.spec.question).run(timeout=60)
        self.assertFalse(app.exception)
        history = app.session_state['conversation'].history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].view.kind, 'answer')
        self.assertGreaterEqual(len(history[0].view.citations), 2)
        self.assertFalse(history[0].view.proposed)
        self.assertEqual(self.evidence['proposed_evidence'], [])
        self.assertTrue(all(h.chunk.status == APPROVED for h in self.hits[0]))
        self.assertTrue(app.chat_input[0].disabled)
        app.run(); self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual(self.submit(turn_id='new-session').code, 'one_shot_spent')
        self.assertEqual(self.sent, ['POST'])
        self.assertNotIn('chromadb', sys.modules)
        self.assertFalse(list((self.root/'.local').glob('chroma-worker-*')))

    def test_draft_cross_policy_citations_and_sibling_evidence_are_preserved(self):
        self.authorize(mode='draft_review', question='An engineer needs temporary production access to troubleshoot an AKS issue. Who should approve it, and which team grants the access?')
        view = self.submit()
        self.assertEqual(view.kind, 'answer')
        self.assertTrue(view.proposed)
        self.assertTrue(all('(proposed)' in c for c in view.citations))
        self.assertGreaterEqual(len({c.document_id for c in view.sources}), 2)
        self.assertTrue(any(hit.context for hit in self.hits[0]))
        for g in self.evidence['proposed_evidence']:
            ids = {s['chunk_id'] for s in g['sources']}
            for source in g['sources']:
                self.assertTrue(set(source['required_context_ids']) <= ids)
        self.assertEqual(self.sent, ['POST'])
        self.assertEqual(view.coverage, 'not_assessed')

    def test_model_authored_evidence_limited_abstention_is_distinct_from_failure(self):
        self.authorize(question='What is the corporate policy for lunar tourism?'); self.behavior = 'abstain'
        view = self.submit()
        self.assertEqual(view.kind, 'abstain')
        self.assertIn('retrieval may be incomplete', view.text)
        self.assertEqual(self.record()['answer_kind'], 'abstain')
        self.assertEqual(self.sent, ['POST'])

    def test_invalid_citation_is_withheld_and_cannot_regenerate(self):
        self.authorize(); self.behavior = 'citation'
        self.assertEqual(self.submit().code, 'response_rejected')
        self.assertEqual(self.record()['status'], 'not_accepted')
        self.assertEqual(self.submit(turn_id='retry').code, 'one_shot_spent')
        self.assertEqual(self.sent, ['POST'])

    def test_unsupported_uncited_statement_is_withheld(self):
        self.authorize(); self.behavior = 'unsupported'
        self.assertEqual(self.submit().code, 'response_rejected')
        self.assertEqual(self.sent, ['POST'])

    def test_invalid_http_envelope_is_not_an_abstention(self):
        self.authorize(); self.behavior = 'envelope'
        self.assertEqual(self.submit().code, 'provider_failed')
        self.assertEqual(json.loads(self.budget.path.read_bytes())['workflows'][0]['status'], 'uncertain')
        self.assertEqual(self.submit(turn_id='retry').code, 'one_shot_spent')
        self.assertEqual(self.sent, ['POST'])

    def test_timeout_keeps_reservation_and_restarted_ui_is_exhausted(self):
        self.authorize(); self.behavior = 'timeout'
        self.assertEqual(self.submit().code, 'provider_failed')
        state = self.budget.inspect_history(diagnostic_approval=self.f.preview.plan_sha256)
        self.assertEqual(state.attempts_consumed, 1)
        self.assertEqual(str(state.reserved_usd), '0.003687')
        self.assertEqual(self.submit(turn_id='restart').code, 'one_shot_spent')
        self.assertEqual(self.sent, ['POST'])

    def test_missing_approval_or_mismatch_has_no_credential_retrieval_or_transport_io(self):
        self.authorize()
        self.assertEqual(self.submit(question='A different question?').code, 'one_shot_mismatch')
        with patch.dict(os.environ, NORTHSTAR_LOCAL_TEST_RUN_APPROVAL='invalid'):
            self.assertEqual(self.submit().code, 'budget_unavailable')
        self.assertFalse(self.opens); self.assertFalse(self.hits); self.assertFalse(self.sent)
        self.assertEqual(json.loads(self.budget.path.read_bytes())['workflows'], [])

    def test_failed_settlement_withholds_answer_and_never_repeats_transport(self):
        self.authorize()
        with patch.object(self.budget, '_settle', side_effect=OSError('synthetic storage failure')):
            self.assertEqual(self.submit().code, 'budget_unavailable')
        self.assertNotEqual(self.record()['status'], 'accepted')
        self.assertEqual(json.loads(self.budget.path.read_bytes())['workflows'][0]['status'], 'reserved')
        self.assertEqual(self.submit(turn_id='restart').code, 'one_shot_spent')
        self.assertEqual(self.sent, ['POST'])


if __name__ == '__main__': unittest.main()
