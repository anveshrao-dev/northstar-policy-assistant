"""Loopback access and durable local execution, using synthetic authority only."""
from contextlib import contextmanager
from datetime import date, timedelta
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from budget_fixture import BILLING, SCOPE, Provider, document, install, request
from checkout_fixture import checkout
from test_result_capture import success
from northstar_atlas import live_access, local_live
from northstar_atlas.live_budget import BudgetError, configured_budget
from northstar_atlas.ui.runtime import ResourceOwner, Submission


class LocalLiveTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        os.environ.update(NORTHSTAR_PROJECT_ROOT=str(self.root), NORTHSTAR_CONTROLLED_LIVE='1',
            NORTHSTAR_UI_ENABLE_LIVE='1', NORTHSTAR_UI_ENABLE_PINECONE='0',
            NORTHSTAR_UI_ENABLE_GRAPH='0', VECTOR_STORE='chroma',
            NORTHSTAR_BUDGET_SCOPE=SCOPE, NORTHSTAR_BILLING_SCOPE=BILLING,
            NORTHSTAR_BUDGET_MAX_ATTEMPTS='3')
        self.enterContext(patch('platform.system', return_value='Darwin'))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        self.path = install(self.root, document())
        self.options = local_live.server_options(8501)
        from streamlit.config import get_option
        self.enterContext(patch('streamlit.config.get_option', side_effect=lambda key:
            self.options[key.replace('.', '_')] if key.replace('.', '_') in self.options else get_option(key)))
        self.headers = [('Host', '127.0.0.1:8501'), ('Origin', 'http://127.0.0.1:8501')]
        self.client = SimpleNamespace(remote_ip='127.0.0.1', headers=self.headers)
        self.client_patch = self.enterContext(patch('streamlit.runtime.context._get_client_context',
                                                   side_effect=lambda: self.client))
        self.private = self.enterContext(patch('northstar_atlas.live_preflight.require_private_preview'))
        self.session = live_access.local_session(self.root, backend='chroma', port=8501,
                                                 enable_graph=False, approved=True)
        self.enterContext(self.session)
        self.budget = configured_budget(self.root)
        self.calls = []
        self.closed = []

    def owner(self, *, fail=None):
        @contextmanager
        def opener(root, **kw):
            flow = json.loads(self.path.read_bytes())['workflows'][-1]
            self.assertEqual((flow['status'], flow['calls']), ('reserved', []))
            self.assertIs(kw['budget'], self.budget)
            self.assertTrue(kw['allow_paid'])
            def ask(question, **args):
                self.calls.append(question)
                kw['reservation'].invoke(Provider(fail), 'generator', 0, request())
                return success()
            try:
                yield SimpleNamespace(ask=ask)
            finally:
                self.closed.append(True)
        owner = ResourceOwner(self.root, live_enabled=True, budget=self.budget, opener=opener)
        self.addCleanup(owner.close)
        return owner

    def submission(self, ident='once', backend='chroma'):
        return Submission(ident, 'A synthetic policy question?', 'approved_only', backend, date.today())

    def test_local_access_does_not_call_hosted_preview(self):
        self.assertIs(self.budget.before_request, live_access.require_local_access)
        self.budget.before_request()
        self.private.assert_not_called()

    def test_no_session_remote_origin_forwarding_and_configuration_drift_deny(self):
        variants = [None, SimpleNamespace(remote_ip='192.0.2.1', headers=self.client.headers),
                    SimpleNamespace(remote_ip='127.0.0.1', headers=[]),
                    SimpleNamespace(remote_ip='127.0.0.1', headers=[self.headers[0], ('Origin', 'https://remote.invalid')]),
                    SimpleNamespace(remote_ip='127.0.0.1', headers=[('Host', 'localhost:8501'), self.headers[1]]),
                    SimpleNamespace(remote_ip='127.0.0.1', headers=self.headers+[('X-Forwarded-For', '127.0.0.1')]),
                    SimpleNamespace(remote_ip='127.0.0.1', headers=self.headers+[self.headers[0]])]
        original = self.client
        for client in variants:
            self.client = client
            with self.subTest(client=client), self.assertRaises(live_access.LocalAccessError):
                live_access.require_local_access()
        self.client = original
        for key, value in [('server_address', '0.0.0.0'), ('server_enableCORS', False),
                           ('server_enableXsrfProtection', False), ('server_port', 8502)]:
            with self.subTest(key=key), patch.dict(self.options, {key: value}), self.assertRaises(live_access.LocalAccessError):
                live_access.require_local_access()
        for key, value in [('REPLIT_DEV_DOMAIN', 'fixture.replit.dev'), ('NORTHSTAR_CONTROLLED_LIVE', '0'),
                           ('VECTOR_STORE', 'pinecone'), ('NORTHSTAR_UI_ENABLE_GRAPH', '1')]:
            with self.subTest(key=key), patch.dict(os.environ, {key: value}), self.assertRaises(live_access.LocalAccessError):
                live_access.require_local_access()

    def test_settlement_then_recording_before_presentation_and_fresh_resources(self):
        owner = self.owner()
        from northstar_atlas.ui import runtime
        project = runtime.project_result
        def check_record(result, mode):
            flow = json.loads(self.path.read_bytes())['workflows'][-1]
            self.assertEqual(flow['status'], 'closed')
            saved = json.loads((self.root/f'.local/linear-demo-results/{flow["id"]}.json').read_bytes())
            self.assertEqual(saved['status'], 'accepted')
            return project(result, mode)
        with patch.object(runtime, 'project_result', side_effect=check_record):
            self.assertEqual(owner.submit(self.submission(), consent=True).kind, 'answer')
            self.assertEqual(owner.submit(self.submission('next'), consent=True).kind, 'answer')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.closed), 2)
        self.assertEqual(owner.submit(self.submission(), consent=True).code, 'duplicate')
        self.assertEqual(self.owner().submit(self.submission(), consent=True).code, 'budget_unavailable')
        self.assertEqual(len(self.calls), 2)

    def test_consent_backend_and_access_fail_before_resources(self):
        owner = self.owner()
        self.assertEqual(owner.submit(self.submission(), consent=False).code, 'opt_in_required')
        self.assertEqual(owner.submit(self.submission(backend='pinecone'), consent=True).code, 'backend_disabled')
        self.client = None
        self.assertEqual(owner.submit(self.submission('no-session'), consent=True).code, 'budget_unavailable')
        self.assertEqual(self.calls, [])
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'], [])

    def test_provider_failure_keeps_reservation_and_never_retries(self):
        owner = self.owner(fail=TimeoutError('synthetic failure'))
        self.assertEqual(owner.submit(self.submission(), consent=True).kind, 'error')
        self.assertEqual(owner.submit(self.submission(), consent=True).code, 'duplicate')
        self.assertEqual(len(self.calls), 1)
        flow = json.loads(self.path.read_bytes())['workflows'][0]
        self.assertEqual(flow['status'], 'uncertain')
        self.assertEqual(str(self.budget.inspect_history().reserved_usd), '0.003687')

    def test_exhausted_missing_corrupt_or_expired_authority_never_calls_provider(self):
        owner = self.owner()
        for i in range(3):
            self.assertEqual(owner.submit(self.submission(f'call-{i}'), consent=True).kind, 'answer')
        self.assertEqual(owner.submit(self.submission('exhausted'), consent=True).code, 'budget_unavailable')
        self.assertEqual(len(self.calls), 3)
        for state in ('missing', 'corrupt', 'expired'):
            with self.subTest(state=state):
                if state == 'missing': self.path.unlink()
                elif state == 'corrupt': self.path.write_text('{}')
                else:
                    from datetime import datetime, timezone
                    value = document()
                    value['pricing_revisions'][0]['valid_until'] = (datetime.now(timezone.utc)-timedelta(days=1)).isoformat()
                    self.path.write_text(json.dumps(value))
                with self.assertRaises(BudgetError):
                    local_live.check_budget(self.root, enable_graph=False, backend='chroma')
                self.assertEqual(len(self.calls), 3)

    def test_launcher_requires_explicit_approval_and_no_hosted_environment(self):
        self.session.__exit__(None, None, None)
        self.assertIsNone(live_access.local_backend())
        with self.assertRaises(live_access.LocalAccessError):
            with live_access.local_session(self.root, backend='chroma', port=8501, enable_graph=False, approved=False):
                self.fail('unauthorized')
        # All other prerequisites are valid: approval is the sole denied input.
        with live_access.local_session(self.root, backend='chroma', port=8501,
                                       enable_graph=False, approved=True):
            with patch.dict(os.environ, REPL_ID='synthetic'), self.assertRaises(live_access.LocalAccessError):
                live_access.require_local_access()

    def test_ssl_configuration_is_rejected_instead_of_changing_local_origin(self):
        with patch.dict(self.options, server_sslCertFile='synthetic-cert'), self.assertRaises(live_access.LocalAccessError):
            live_access.require_local_server()

    def test_settlement_failure_and_interruption_never_publish_or_replay(self):
        owner = self.owner()
        with patch.object(self.budget, '_settle', side_effect=OSError('synthetic write failure')):
            self.assertEqual(owner.submit(self.submission(), consent=True).kind, 'error')
        self.assertEqual(self.owner().submit(self.submission(), consent=True).code, 'budget_unavailable')
        self.assertEqual(len(self.calls), 1)
        record = json.loads((self.root/'.local/linear-demo-results/once.json').read_bytes())
        self.assertEqual(record['status'], 'outcome_unknown')
        interrupted = self.owner(fail=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            interrupted.submit(self.submission('interrupt'), consent=True)
        self.assertEqual(interrupted.submit(self.submission('interrupt'), consent=True).code, 'duplicate')
        flow = json.loads(self.path.read_bytes())['workflows'][-1]
        self.assertEqual(flow['status'], 'uncertain')
        self.assertEqual(len(self.calls), 2)

    def test_streamlit_local_selection_consent_and_duplicate_rerun(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        original = sys.modules['__main__']
        self.addCleanup(sys.modules.__setitem__, '__main__', original)
        st.cache_resource.clear(); self.addCleanup(st.cache_resource.clear)
        owner = self.owner()
        with patch.object(Path, 'cwd', return_value=self.root), patch('northstar_atlas.ui.runtime.ResourceOwner', return_value=owner):
            app = AppTest.from_file(str(self.root/'src/northstar_atlas/ui/app.py')).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='experience').value, 'Offline demo')
            app.radio(key='experience').set_value('Live Q&A').run()
            self.assertEqual(app.selectbox(key='workflow').value, 'standard')
            self.assertEqual(app.selectbox(key='backend').options, ['Chroma · local'])
            self.assertTrue(app.chat_input[0].disabled)
            app.checkbox(key='paid_opt_in').check().run()
            app.chat_input[0].set_value('An authored test question?').run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['conversation'].history[-1].view.kind, 'answer')
            app.run()
            self.assertEqual(len(self.calls), 1)


class LocalBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        os.environ.update(NORTHSTAR_PROJECT_ROOT=str(self.root), NORTHSTAR_BUDGET_SCOPE=SCOPE,
                          NORTHSTAR_BILLING_SCOPE=BILLING, NORTHSTAR_BUDGET_MAX_ATTEMPTS='3')
        self.enterContext(patch('platform.system', return_value='Darwin'))
        self.enterContext(patch.object(Path, 'cwd', return_value=self.root))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        self.path = install(self.root, document())

    def test_missing_project_temp_directory_stops_before_bootstrap(self):
        from streamlit.web import bootstrap
        with patch.object(bootstrap, 'run') as run:
            self.assertEqual(local_live.main(['--approve-local-live']), 1)
            run.assert_not_called()

    def test_missing_project_cache_directory_stops_before_bootstrap(self):
        from streamlit.web import bootstrap
        (self.root/'.local/tmp').mkdir()
        (self.root/'.venv/bin').mkdir(parents=True)
        with patch.object(bootstrap, 'run') as run:
            self.assertEqual(local_live.main(['--approve-local-live']), 1)
            run.assert_not_called()

    def test_exact_bootstrap_options_without_provider_io_or_ledger_mutation(self):
        from streamlit.web import bootstrap
        # Use the pinned real config parser; only replace the blocking HTTP server.
        import streamlit.config as config
        old = config._config_options
        self.addCleanup(setattr, config, '_config_options', old)
        for relative in ('.local/tmp', '.local/cache', '.venv/bin'):
            (self.root/relative).mkdir(parents=True, exist_ok=True)
        original = self.path.read_bytes()
        def start(path, hello, args, flags):
            self.assertEqual(path, str(self.root/'src/northstar_atlas/ui/app.py'))
            self.assertEqual((hello, args), (False, []))
            live_access.require_local_server()
            self.assertEqual(live_access.local_backend(), 'chroma')
            self.assertIs(configured_budget(self.root).before_request, live_access.require_local_access)
        with patch.object(bootstrap, 'run', side_effect=start) as run:
            self.assertEqual(local_live.main(['--approve-local-live']), 0)
            run.assert_called_once()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertIsNone(live_access.local_backend())
        self.assertNotIn('NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED', os.environ)

    def test_default_access_stays_hosted_and_env_cannot_select_local_policy(self):
        with patch.dict(os.environ, NORTHSTAR_LOCAL_LIVE_APPROVED='1'):
            from northstar_atlas.live_preflight import require_private_preview
            self.assertIs(configured_budget(self.root).before_request, require_private_preview)

    def test_denied_startup_never_runs_server_or_creates_ledger(self):
        from streamlit.web import bootstrap
        for relative in ('.local/tmp', '.local/cache', '.venv/bin'):
            (self.root/relative).mkdir(parents=True, exist_ok=True)
        self.assertIsNone(live_access.local_backend())
        self.path.unlink()
        with patch.object(bootstrap, 'run') as run, patch.object(
                local_live, 'check_budget', wraps=local_live.check_budget) as checked:
            self.assertEqual(local_live.main(['--approve-local-live']), 1)
            checked.assert_called_once_with(self.root, enable_graph=False, backend='chroma')
            run.assert_not_called()
        self.assertFalse(self.path.exists())
        self.assertIsNone(live_access.local_backend())


class LocalExistingPipelineTests(unittest.TestCase):
    def configure(self, fixture, *, backend, graph):
        os.environ.update(VECTOR_STORE=backend, NORTHSTAR_UI_ENABLE_LIVE='1',
                          NORTHSTAR_UI_ENABLE_PINECONE=str(int(backend == 'pinecone')),
                          NORTHSTAR_UI_ENABLE_GRAPH=str(int(graph)),
                          NORTHSTAR_BUDGET_SCOPE=fixture.budget.scope,
                          NORTHSTAR_BILLING_SCOPE=fixture.budget.billing_scope,
                          NORTHSTAR_BUDGET_MAX_ATTEMPTS=str(fixture.budget.policy.max_attempts))
        for name in live_access._HOSTED_ENV:
            os.environ.pop(name, None)
        self.enterContext(patch('platform.system', return_value='Darwin'))
        self.enterContext(live_access.local_session(fixture.root, backend=backend, port=8501,
                                                   enable_graph=graph, approved=True))
        options = local_live.server_options(8501)
        self.enterContext(patch('streamlit.config.get_option', side_effect=lambda k: options.get(k.replace('.', '_'))))
        self.enterContext(patch('streamlit.runtime.context._get_client_context', return_value=SimpleNamespace(
            remote_ip='127.0.0.1', headers=[('Host', '127.0.0.1:8501'), ('Origin', 'http://127.0.0.1:8501')])))
        fixture.budget.before_request = live_access.require_local_access

    def test_existing_pinecone_batch_preserves_approvals_five_calls_and_settlement(self):
        import test_demo_batch as batch_fixtures
        fixture = batch_fixtures.BatchTests('runTest'); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        preview, owner = fixture.enable_runtime()
        self.configure(fixture, backend='pinecone', graph=False)
        self.assertEqual(local_live.check_budget(fixture.root, enable_graph=False, backend='pinecone').policy.max_attempts, 7)
        for i in range(5):
            result = owner.submit(fixture.submission(f'local-batch-{i}'), consent=True)
            self.assertEqual(result.kind, 'answer', result)
        self.assertEqual(owner.submit(fixture.submission('sixth'), consent=True).code, 'batch_exhausted')
        self.assertEqual(len(fixture.sent), 5)

    def test_existing_seven_call_graph_uses_local_access_and_both_backends(self):
        import test_live_graph as graph_fixtures
        # Each profile run gets its own synthetic authority and isolated scope.
        from contextlib import ExitStack
        for backend in ('chroma', 'pinecone'):
            with self.subTest(backend=backend), ExitStack() as stack:
                case = LocalExistingPipelineTests('runTest')
                stack.callback(case.doCleanups)
                fixture = graph_fixtures.LiveGraphTests('runTest'); fixture.setUp()
                stack.callback(fixture.doCleanups)
                fixture.wire(judgments=('uncertain', 'relevant'), semantics=('incomplete', 'supported'), backend=backend)
                case.configure(fixture, backend=backend, graph=True)
                result = fixture.owner(pinecone_enabled=backend == 'pinecone').submit(fixture.request(backend=backend), consent=True)
                self.assertEqual(result.kind, 'answer', result)
                self.assertEqual(fixture.sent, ['router', 'grader', 'grader', 'generator', 'semantic_checker', 'generator', 'semantic_checker'])
                self.assertEqual(fixture.budget.inspect_history().attempts_consumed, 7)
                self.assertEqual(len(fixture.closed), 1)
                fixture.access.assert_not_called()


if __name__ == '__main__':
    unittest.main()
