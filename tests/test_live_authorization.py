"""No paid path without a durable permit; synthetic state and inert I/O only."""
from contextlib import contextmanager, redirect_stderr
from datetime import date
import io
import os
import unittest
from unittest.mock import Mock, patch

import httpx

from budget_fixture import request, controlled_budget
from checkout_fixture import checkout
from northstar_atlas.nebius import NebiusProvider, NebiusSettings, ProviderError
from northstar_atlas.qa import QAService, QAConfigurationError, open_qa_service
from northstar_atlas.ui.runtime import ResourceOwner, Submission
from test_ui import success


class LiveAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, {'NORTHSTAR_PROJECT_ROOT': str(self.root),
                                                'NORTHSTAR_UI_ENABLE_LIVE': '1'}, clear=True))

    def test_direct_ui_rejects_absent_and_disabled_controlled_mode_before_opener(self):
        calls = []
        @contextmanager
        def opener(*args, **kwargs):
            calls.append(kwargs)
            yield Mock(ask=Mock(return_value=success()))
        for flag in (None, '0'):
            if flag is not None:
                os.environ['NORTHSTAR_CONTROLLED_LIVE'] = flag
            with self.subTest(flag=flag):
                owner = ResourceOwner(self.root, live_enabled=True, opener=opener)
                try:
                    result = owner.submit(Submission('one', 'Who?', 'approved_only', 'chroma', date.today()), consent=True)
                    self.assertEqual(result.kind, 'error')
                    self.assertEqual(calls, [])
                finally:
                    owner.close()

    def test_paid_flag_and_explicit_budget_opt_out_cannot_send_http(self):
        sent = []
        def send(req):
            sent.append(req)
            return httpx.Response(200, json={'model': request().model, 'choices': [
                {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{}'}}]})
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            requires_budget=False, transport=httpx.MockTransport(send)) as provider:
            with self.assertRaises(ProviderError):
                provider.generate(request())
        self.assertEqual(sent, [])

    def test_paid_service_opener_without_budget_or_permit_denies_before_credentials(self):
        with patch('northstar_atlas.nebius.load_settings') as credentials, \
             patch('northstar_atlas.qa.open_retriever') as retrieval:
            with self.assertRaises(QAConfigurationError):
                with open_qa_service(self.root, allow_paid=True):
                    self.fail('unreserved service')
        credentials.assert_not_called()
        retrieval.assert_not_called()

    def test_graph_opener_cannot_create_paid_provider_without_reservation(self):
        from northstar_atlas.orchestration import open_graph_qa_service
        with patch('northstar_atlas.nebius.load_settings') as credentials, \
             patch('northstar_atlas.orchestration.load_catalog') as catalog:
            with self.assertRaises(QAConfigurationError):
                with open_graph_qa_service(self.root, snapshot_id='a'*64, allow_paid=True):
                    self.fail('unreserved graph')
        credentials.assert_not_called(); catalog.assert_not_called()

    def test_direct_graph_paid_provider_cannot_retrieve_or_route_without_authority(self):
        from test_orchestration import OrchestrationTests
        from northstar_atlas.orchestration import SEMANTIC_PROFILE
        fixture = OrchestrationTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            transport=httpx.MockTransport(lambda _: self.fail('provider I/O'))) as provider:
            for profile in ('bounded_v1', 'llm_routing_grading_v1', SEMANTIC_PROFILE):
                service, first, second, _ = fixture.make(provider=provider)
                service._profile = profile
                result = fixture.ask(service)
                self.assertFalse(result.accepted)
                self.assertEqual(result.public.code, 'configuration_failed')
                self.assertEqual(first.calls + second.calls, [])


class LivePermitTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.budget = controlled_budget(self, self.root, calls=8)

    def spec(self, **changes):
        from northstar_atlas.live_budget import WorkflowRequest, LINEAR
        values = dict(invocation_id='one', profile=LINEAR, question='Who?', mode='approved_only',
                      backend='chroma', reference_date=date.today())
        values.update(changes)
        return WorkflowRequest(**values)

    def test_missing_invalid_wrong_coordinator_backend_profile_and_closed_permits(self):
        from northstar_atlas.live_budget import GRAPH
        other_root = checkout(self)
        other = controlled_budget(self, other_root, calls=8)
        # Restore the authorized app root after constructing an unrelated fixture.
        os.environ['NORTHSTAR_PROJECT_ROOT'] = str(self.root)
        # Both fixture coordinators must use the currently selected verifier.
        self.budget.before_request = other.before_request
        with self.budget.reserve(self.spec()) as permit:
            cases = [(None, permit, 'chroma'), (self.budget, None, 'chroma'),
                     (self.budget, object(), 'chroma'), (other, permit, 'chroma'),
                     (self.budget, permit, 'pinecone')]
            for budget, reservation, backend in cases:
                with self.subTest(case=(type(budget).__name__, type(reservation).__name__, backend)), \
                     patch('northstar_atlas.nebius.load_settings') as credentials, \
                     patch('northstar_atlas.qa.open_retriever') as retrieval:
                    with self.assertRaises(QAConfigurationError):
                        with open_qa_service(self.root, backend=backend, allow_paid=True,
                                             budget=budget, reservation=reservation): self.fail('opened')
                    credentials.assert_not_called(); retrieval.assert_not_called()
        for reservation in (permit,):
            with self.assertRaises(QAConfigurationError):
                with open_qa_service(self.root, allow_paid=True, budget=self.budget, reservation=reservation):
                    self.fail('expired permit')
        with self.budget.reserve(self.spec(invocation_id='graph', profile=GRAPH)) as graph:
            with patch('northstar_atlas.nebius.load_settings') as credentials:
                with self.assertRaises(QAConfigurationError):
                    with open_qa_service(self.root, allow_paid=True, budget=self.budget, reservation=graph):
                        self.fail('wrong profile')
                credentials.assert_not_called()

    def test_request_binding_and_missing_direct_qa_permit_fail_before_search(self):
        store = Mock()
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            transport=httpx.MockTransport(lambda _: self.fail('provider I/O'))) as provider:
            service = QAService(store, None, provider, corpus_fingerprint='fixture',
                                backend='chroma', budget=self.budget)
            self.assertEqual(service.ask('Who?').public.code, 'configuration_failed')
            with self.budget.reserve(self.spec()) as permit:
                for question, mode, backend in (('Different?', 'approved_only', 'chroma'),
                                                ('Who?', 'draft_review', 'chroma'),
                                                ('Who?', 'approved_only', 'pinecone')):
                    service._backend = backend
                    result = service.ask(question, mode=mode, reservation=permit, reference_date=date.today())
                    self.assertEqual(result.public.code, 'configuration_failed')
            store.search.assert_not_called()

    def test_graph_injected_paid_provider_rechecks_access_before_resources(self):
        from northstar_atlas.orchestration import open_graph_qa_service, SEMANTIC_PROFILE
        from northstar_atlas.live_budget import GRAPH
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            transport=httpx.MockTransport(lambda _: self.fail('provider I/O'))) as provider:
            with self.budget.reserve(self.spec(profile=GRAPH)) as permit:
                self.budget.before_request.side_effect = ValueError('private access denied')
                with patch('northstar_atlas.orchestration.load_catalog') as catalog, \
                     patch('northstar_atlas.orchestration.open_retriever') as retrieval:
                    with self.assertRaises(QAConfigurationError):
                        with open_graph_qa_service(self.root, backend='chroma', profile=SEMANTIC_PROFILE,
                                snapshot_id='a'*64, provider=provider, budget=self.budget, reservation=permit):
                            self.fail('access bypass')
                    catalog.assert_not_called(); retrieval.assert_not_called()

    def test_access_probe_expiry_stops_direct_qa_before_search(self):
        from datetime import timedelta
        store = Mock()
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            transport=httpx.MockTransport(lambda _: self.fail('provider I/O'))) as provider:
            service = QAService(store, None, provider, corpus_fingerprint='fixture', backend='chroma', budget=self.budget)
            with self.budget.reserve(self.spec()) as permit:
                now = self.budget.clock()
                self.budget.before_request.side_effect = lambda: setattr(self.budget, 'clock', lambda: now + timedelta(hours=2))
                result = service.ask('Who?', reference_date=date.today(), reservation=permit)
                self.assertEqual(result.public.code, 'configuration_failed')
                store.search.assert_not_called()

    def test_direct_paid_graph_checks_access_before_any_node(self):
        from northstar_atlas.orchestration import GraphQAService, SEMANTIC_PROFILE
        from northstar_atlas.live_budget import GRAPH
        from test_bm25 import catalog
        from test_context import chunk
        from test_orchestration_semantic import ScopedStore
        store = ScopedStore()
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            transport=httpx.MockTransport(lambda _: self.fail('provider I/O'))) as provider:
            service = GraphQAService({'dense': store}, None, provider, catalog=catalog([chunk()]),
                strategy_plan=('dense',), backend='chroma', profile=SEMANTIC_PROFILE, budget=self.budget)
            with self.budget.reserve(self.spec(profile=GRAPH)) as permit:
                for error in (ValueError('private access denied'), OSError('private path')):
                    self.budget.before_request.side_effect = error
                    with patch.object(service._graph, 'invoke') as invoke:
                        result = service.ask('Who?', reference_date=date.today(), reservation=permit)
                    self.assertEqual(result.public.code, 'configuration_failed')
                    self.assertNotIn('private', result.public.message)
                    invoke.assert_not_called(); self.assertEqual(store.calls, [])

    def test_unauthorized_access_callback_cannot_authorize_transport(self):
        from northstar_atlas.nebius import ProviderError
        sent = []
        self.budget.before_request = lambda: None
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            transport=httpx.MockTransport(lambda r: sent.append(r))) as provider:
            with self.budget.reserve(self.spec()) as permit:
                with self.assertRaises(ProviderError):
                    permit.invoke(provider, 'generator', 0, request())
        self.assertEqual(sent, [])

    def test_hosted_ui_records_after_settlement_before_projection(self):
        import json
        from northstar_atlas.ui import runtime
        from budget_fixture import Provider
        calls = []
        @contextmanager
        def opener(root, **kw):
            def ask(question, **options):
                calls.append(question)
                options['reservation'].invoke(Provider(), 'generator', 0, request())
                return success()
            yield Mock(ask=ask)
        owner = ResourceOwner(self.root, live_enabled=True, budget=self.budget, opener=opener)
        self.addCleanup(owner.close)
        project = runtime.project_result
        def after_record(result, mode):
            state = json.loads(self.budget.path.read_bytes())
            self.assertEqual(state['workflows'][-1]['status'], 'closed')
            saved = json.loads((self.root/'.local/linear-demo-results/one.json').read_bytes())
            self.assertEqual(saved['status'], 'accepted')
            return project(result, mode)
        submission = Submission('one', 'Who?', 'approved_only', 'chroma', date.today())
        with patch.object(runtime, 'project_result', side_effect=after_record):
            self.assertEqual(owner.submit(submission, consent=True).kind, 'answer')
        self.assertEqual(owner.submit(submission, consent=True).code, 'duplicate')
        self.assertEqual(calls, ['Who?'])

    def test_direct_qa_paid_provider_denies_before_retrieval(self):
        store = Mock()
        with NebiusProvider(NebiusSettings('synthetic-test-key'), allow_paid=True,
                            transport=httpx.MockTransport(lambda _: self.fail('provider I/O'))) as provider:
            result = QAService(store, None, provider, corpus_fingerprint='fixture', backend='chroma').ask('Who?')
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, 'configuration_failed')
        store.search.assert_not_called()

    def test_legacy_cli_paid_flag_denies_before_credentials_or_resources(self):
        from northstar_atlas import answers
        with patch.object(answers, 'load_settings') as credentials, \
             patch.object(answers, 'open_retriever') as retrieval, \
             patch.object(answers.Path, 'cwd', return_value=self.root), redirect_stderr(io.StringIO()):
            self.assertEqual(answers.main(['ask', 'Who?', '--allow-paid']), 1)
        credentials.assert_not_called()
        retrieval.assert_not_called()
