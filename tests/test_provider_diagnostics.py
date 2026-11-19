"""Private failure categories with synthetic budgets and mocked HTTP only."""
from datetime import date
from decimal import Decimal
import io
import json
import logging
import multiprocessing
import os
import socket
import stat
import unittest
from unittest.mock import patch

import httpx

from budget_fixture import BILLING, SCOPE, Provider, document, install, request
from checkout_fixture import checkout
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, GRAPH, LINEAR
from northstar_atlas.llm import MODEL_ID
from northstar_atlas.nebius import NebiusProvider, NebiusSettings, ProviderError


SECRET = 'synthetic-secret-never-record'
PRIVATE = 'private prompt response header and exception payload'


def envelope():
    return {'model': MODEL_ID,
            'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{}'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}


class ProviderDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root), NORTHSTAR_CONTROLLED_LIVE='1',
            NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1', NORTHSTAR_UI_ENABLE_GRAPH='1'))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        self.path = install(self.root, document(calls=8))
        self.diagnostic = self.path.with_name('provider-failure.json')
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
                                      policy=BudgetPolicy(max_attempts=8), before_request=self.enterContext(
                                          patch('northstar_atlas.live_preflight.require_private_preview')))
        self.sent = []

    def invoke(self, outcome, *, ident='one', profile=GRAPH, allow_paid=True, mutate=None):
        def send(req):
            self.sent.append(req.method)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        with NebiusProvider(NebiusSettings(SECRET), allow_paid=allow_paid, requires_budget=True,
                            transport=httpx.MockTransport(send)) as provider:
            if mutate:
                mutate(provider)
            spec = WorkflowRequest(ident, profile, 'synthetic question', 'draft_review', 'chroma', date.today())
            with self.budget.reserve(spec) as permit:
                req = request(256, 8192) if profile == GRAPH else request()
                return permit.invoke(provider, 'router' if profile == GRAPH else 'generator', 0, req)

    def record(self, category, attempted=True, status=200, stage='router', timeout_subtype='read_timeout'):
        self.assertTrue(self.diagnostic.is_file(), 'sanitized failure diagnostic was not retained')
        raw = self.diagnostic.read_bytes()
        self.assertLessEqual(len(raw), 512)
        value = json.loads(raw)
        fields = {'version', 'category', 'stage', 'ordinal', 'transport_attempted', 'http_status', 'started_at'}
        if category == 'transport_timeout':
            fields.add('timeout_subtype')
            self.assertEqual(value.get('timeout_subtype'), timeout_subtype)
        self.assertEqual(set(value), fields)
        self.assertEqual((value['version'], value['category'], value['stage'], value['ordinal'],
                          value['transport_attempted'], value['http_status']),
                         (1, category, stage, 0, attempted, status))
        for forbidden in (SECRET, PRIVATE, 'Authorization', 'Bearer', 'synthetic question'):
            self.assertNotIn(forbidden, raw.decode())
        self.assertEqual(stat.S_IMODE(self.diagnostic.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.diagnostic.parent.stat().st_mode), 0o700)
        return value

    def uncertain(self):
        flow = json.loads(self.path.read_bytes())['workflows'][-1]
        self.assertEqual(flow['status'], 'uncertain')
        self.assertEqual(flow['calls'][0]['status'], 'failed_or_uncertain')
        self.assertIsNone(flow['calls'][0]['usage'])
        state = self.budget.inspect_history()
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (7, Decimal('0.024629')))

    def test_timeout_is_retained_without_retry_or_refund(self):
        with self.assertRaisesRegex(ProviderError, 'provider timeout'):
            self.invoke(httpx.ReadTimeout(PRIVATE + SECRET))
        self.record('transport_timeout', status=None)
        self.assertEqual(self.sent, ['POST'])
        self.uncertain()

    def check_timeout_subtype(self, exception_type, expected):
        sensitive = 'https://private.invalid/request?token=' + SECRET
        request = httpx.Request('POST', sensitive, headers={'Authorization': 'Bearer ' + SECRET},
                               content=PRIVATE.encode())
        # Deliberately misleading text: classification must use the exception type.
        error = exception_type('ReadTimeout connect_timeout ' + sensitive + PRIVATE, request=request)
        with self.assertRaises(ProviderError) as caught:
            self.invoke(error, profile=LINEAR)
        self.record('transport_timeout', status=None, stage='generator', timeout_subtype=expected)
        self.assertEqual(str(caught.exception), 'provider timeout; no automatic retry, usage may be unknown')
        self.assertEqual(self.sent, ['POST'])
        flow = json.loads(self.path.read_bytes())['workflows'][-1]
        self.assertEqual((flow['status'], len(flow['calls']), flow['calls'][0]['status']),
                         ('uncertain', 1, 'failed_or_uncertain'))
        self.assertIsNone(flow['calls'][0]['usage'])
        state = self.budget.inspect_history()
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (1, Decimal('0.003687')))
        self.assertFalse(self.budget.pending_path.exists())
        self.assertNotIn(sensitive, self.diagnostic.read_text())

    def test_connect_timeout_classification_preserves_linear_accounting(self):
        self.check_timeout_subtype(httpx.ConnectTimeout, 'connect_timeout')

    def test_read_timeout_classification_preserves_linear_accounting(self):
        self.check_timeout_subtype(httpx.ReadTimeout, 'read_timeout')

    def test_write_timeout_classification_preserves_linear_accounting(self):
        self.check_timeout_subtype(httpx.WriteTimeout, 'write_timeout')

    def test_pool_timeout_classification_preserves_linear_accounting(self):
        self.check_timeout_subtype(httpx.PoolTimeout, 'pool_timeout')

    def test_generic_timeout_does_not_classify_from_sensitive_message(self):
        self.check_timeout_subtype(httpx.TimeoutException, 'unknown_timeout')

    def test_timeout_subclass_uses_observable_httpx_type(self):
        class CustomReadTimeout(httpx.ReadTimeout):
            pass
        self.check_timeout_subtype(CustomReadTimeout, 'read_timeout')

    def test_pool_timeout_before_dispatch_does_not_fabricate_transport(self):
        def mutate(provider):
            self.enterContext(patch.object(provider._client, 'request', side_effect=httpx.PoolTimeout(PRIVATE)))
        with self.assertRaises(ProviderError):
            self.invoke(None, profile=LINEAR, mutate=mutate)
        self.record('transport_timeout', attempted=False, status=None, stage='generator', timeout_subtype='pool_timeout')
        self.assertFalse(self.sent)
        self.assertEqual(self.budget.inspect_history().attempts_consumed, 1)

    def test_serializer_rejects_untrusted_timeout_metadata_without_stringifying(self):
        from northstar_atlas.provider_diagnostics import FailureCategory, TransportObservation, failure_bytes
        class Unsafe:
            def __str__(self):
                raise AssertionError('must not stringify untrusted metadata')
        for subtype in (None, 'read_timeout', PRIVATE + SECRET, {'header': SECRET}, Unsafe()):
            with self.subTest(kind=type(subtype).__name__):
                error = ProviderError(PRIVATE + SECRET, category=FailureCategory.TIMEOUT)
                error.timeout_subtype = subtype
                raw = failure_bytes(error, TransportObservation(True, None), stage='generator', ordinal=0,
                                    started_at='2026-01-01T00:00:00+00:00', fallback=FailureCategory.INTERNAL)
                self.assertEqual(json.loads(raw).get('timeout_subtype'), 'unknown_timeout')
                self.assertNotIn(PRIVATE.encode(), raw)
                self.assertNotIn(SECRET.encode(), raw)
                self.assertLessEqual(len(raw), 512)

    def test_non_timeout_diagnostic_ignores_attached_subtype(self):
        from northstar_atlas.provider_diagnostics import FailureCategory, TransportObservation, failure_bytes
        error = ProviderError(PRIVATE, category=FailureCategory.HTTP)
        error.timeout_subtype = SECRET
        raw = failure_bytes(error, TransportObservation(True, 429), stage='generator', ordinal=0,
                            started_at='2026-01-01T00:00:00+00:00', fallback=FailureCategory.INTERNAL)
        self.assertNotIn('timeout_subtype', json.loads(raw))
        self.assertNotIn(SECRET.encode(), raw)

    def test_connection_error_has_no_invented_http_status(self):
        with self.assertRaisesRegex(ProviderError, 'provider transport failure'):
            self.invoke(httpx.ConnectError(PRIVATE + SECRET))
        self.record('connection_error', status=None)
        self.assertEqual(self.sent, ['POST'])

    def test_other_http_transport_error_is_distinct(self):
        with self.assertRaises(ProviderError):
            self.invoke(httpx.RemoteProtocolError(PRIVATE))
        self.record('transport_error', status=None)

    def test_http_rejection_preserves_only_safe_numeric_status(self):
        with self.assertRaisesRegex(ProviderError, 'HTTP 429'):
            self.invoke(httpx.Response(429, text=PRIVATE + SECRET, headers={'Retry-After': '0'}))
        self.record('http_rejection', status=429)
        self.assertEqual(self.sent, ['POST'])
        self.uncertain()

    def test_invalid_json_is_response_envelope_failure(self):
        with self.assertRaisesRegex(ProviderError, 'invalid or unsafe provider response'):
            self.invoke(httpx.Response(200, text=PRIVATE))
        self.record('response_envelope_validation')

    def test_incomplete_finish_state_is_response_envelope_failure(self):
        value = envelope(); value['choices'][0]['finish_reason'] = 'length'
        with self.assertRaisesRegex(ProviderError, 'model response, finish state or usage failed validation'):
            self.invoke(httpx.Response(200, json=value))
        self.record('response_envelope_validation')

    def test_invalid_usage_is_distinct_and_preserves_reservation(self):
        value = envelope(); value['usage']['total_tokens'] = 99
        with self.assertRaisesRegex(ProviderError, 'model response, finish state or usage failed validation'):
            self.invoke(httpx.Response(200, json=value))
        self.record('usage_validation')
        self.uncertain()

    def test_unknown_usage_is_rejected_by_budget_and_classified(self):
        value = envelope(); value.pop('usage')
        with self.assertRaises(BudgetError):
            self.invoke(httpx.Response(200, json=value))
        self.record('usage_validation')
        self.uncertain()

    def test_unexpected_internal_error_records_no_exception_text(self):
        with self.assertRaises(RuntimeError):
            self.invoke(RuntimeError(PRIVATE + SECRET))
        self.record('unexpected_internal', status=None)
        self.uncertain()

    def test_permit_claim_is_not_an_http_attempt(self):
        with self.assertRaisesRegex(ProviderError, 'paid inference is disabled'):
            self.invoke(httpx.Response(200, json=envelope()), allow_paid=False)
        self.record('request_validation', attempted=False, status=None)
        self.assertFalse(self.sent)
        self.uncertain()

    def test_failure_before_transport_dispatch_is_not_reported_as_attempt(self):
        def mutate(provider):
            self.enterContext(patch.object(provider._client, 'request', side_effect=RuntimeError(PRIVATE)))
        with self.assertRaises(RuntimeError):
            self.invoke(httpx.Response(200, json=envelope()), mutate=mutate)
        self.record('unexpected_internal', attempted=False, status=None)
        self.assertFalse(self.sent)

    def test_read_timeout_after_headers_retains_observed_status(self):
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                raise httpx.ReadTimeout(PRIVATE)
                yield b''
        with self.assertRaises(ProviderError):
            self.invoke(httpx.Response(200, stream=BrokenStream()))
        self.record('transport_timeout', status=200)

    def test_secret_echo_and_logs_never_enter_diagnostic(self):
        value = envelope(); value['choices'][0]['message']['content'] = json.dumps({'echo': SECRET})
        logs = io.StringIO(); handler = logging.StreamHandler(logs)
        logger = logging.getLogger('httpx'); logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        with self.assertRaises(ProviderError) as error:
            self.invoke(httpx.Response(200, json=value))
        self.record('response_envelope_validation')
        self.assertNotIn(SECRET, str(error.exception) + logs.getvalue())

    def test_latest_only_storage_is_bounded_and_shared_with_linear(self):
        for ident, failure, category in [('first', httpx.ReadTimeout(PRIVATE), 'transport_timeout'),
                                         ('second', httpx.ConnectError(PRIVATE), 'connection_error')]:
            with self.assertRaises(ProviderError):
                self.invoke(failure, ident=ident, profile=LINEAR)
            self.record(category, status=None, stage='generator')
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()),
                         ['budget.json', 'budget.lock', 'provider-failure.json'])
        self.assertEqual(self.budget.inspect_history().attempts_consumed, 2)

    def test_diagnostic_write_failure_does_not_mask_error_or_release_reservation(self):
        # Reject the sidecar path safely, without following or overwriting its target.
        target = self.root / 'sentinel'; target.write_bytes(b'unchanged')
        self.diagnostic.symlink_to(target)
        with self.assertRaisesRegex(ProviderError, 'provider timeout'):
            self.invoke(httpx.ReadTimeout(PRIVATE))
        self.assertEqual(target.read_bytes(), b'unchanged')
        self.assertTrue(self.diagnostic.is_symlink())
        self.uncertain()

    def test_fake_provider_does_not_fabricate_transport_observation(self):
        spec = WorkflowRequest('fake', GRAPH, 'Who?', 'draft_review', 'chroma', date.today())
        with self.budget.reserve(spec) as permit:
            with self.assertRaises(RuntimeError):
                permit.invoke(Provider(RuntimeError(PRIVATE)), 'router', 0, request(256, 8192))
        self.record('unexpected_internal', attempted=None, status=None)

    def test_success_does_not_write_failure_or_change_accounting(self):
        self.invoke(httpx.Response(200, json=envelope()))
        self.assertFalse(self.diagnostic.exists())
        self.assertEqual(self.sent, ['POST'])
        self.assertEqual(self.budget.inspect_history().attempts_consumed, 1)

    def test_failed_diagnostic_replace_preserves_original_error_and_accounting(self):
        original = os.replace
        def replace(source, destination):
            if destination == self.diagnostic:
                raise OSError(PRIVATE + SECRET)
            return original(source, destination)
        with patch('northstar_atlas.live_budget.os.replace', side_effect=replace):
            with self.assertRaisesRegex(ProviderError, 'provider timeout'):
                self.invoke(httpx.ReadTimeout(PRIVATE))
        self.assertFalse(self.diagnostic.exists())
        self.assertFalse(self.diagnostic.with_suffix('.pending').exists())
        self.uncertain()

    def test_crash_pending_file_blocks_only_diagnostics_without_growth(self):
        pending = self.diagnostic.with_suffix('.pending')
        pending.write_bytes(b'prior sanitized diagnostic'); pending.chmod(0o600)
        with self.assertRaises(ProviderError):
            self.invoke(httpx.ReadTimeout(PRIVATE))
        self.assertEqual(pending.read_bytes(), b'prior sanitized diagnostic')
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()),
                         ['budget.json', 'budget.lock', 'provider-failure.pending'])
        self.uncertain()

    def test_unsafe_existing_sidecar_is_not_overwritten(self):
        self.diagnostic.write_bytes(b'prior'); self.diagnostic.chmod(0o644)
        with self.assertRaises(ProviderError):
            self.invoke(httpx.ReadTimeout(PRIVATE))
        self.assertEqual(self.diagnostic.read_bytes(), b'prior')
        self.uncertain()

    def test_external_authority_receives_diagnostic_with_the_shared_ledger(self):
        authority = checkout(self)
        original_path = self.path; original = self.path.read_bytes()
        self.path = install(authority, document(calls=8))
        self.diagnostic = self.path.with_name('provider-failure.json')
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
            policy=BudgetPolicy(max_attempts=8), authority_root=authority, authorized_authority_root=authority,
            before_request=self.budget.before_request)
        with self.assertRaises(ProviderError):
            self.invoke(httpx.ReadTimeout(PRIVATE))
        self.record('transport_timeout', status=None)
        self.assertEqual(original_path.read_bytes(), original)
        self.assertFalse(original_path.with_name('provider-failure.json').exists())

    def test_transport_observation_does_not_leak_between_invocations(self):
        with self.assertRaises(ProviderError):
            self.invoke(httpx.ReadTimeout(PRIVATE), ident='sent', profile=LINEAR)
        self.record('transport_timeout', status=None, stage='generator')
        with self.assertRaises(ProviderError):
            self.invoke(httpx.Response(200, json=envelope()), ident='not-sent', profile=LINEAR, allow_paid=False)
        self.record('request_validation', attempted=False, status=None, stage='generator')
        self.assertEqual(self.sent, ['POST'])

    def test_internal_bug_in_usage_checker_is_not_a_provider_usage_rejection(self):
        with patch('northstar_atlas.live_budget._validate_usage', side_effect=RuntimeError(PRIVATE)):
            # Patch is limited to the invocation; settlement revalidates normal history.
            with self.assertRaises(RuntimeError):
                self.invoke(httpx.Response(200, json=envelope()))
        self.record('unexpected_internal')
        self.uncertain()

    def test_fifo_sidecar_cannot_hang_failure_settlement(self):
        os.mkfifo(self.diagnostic, 0o600)
        def invoke_in_child():
            try:
                self.invoke(httpx.ReadTimeout(PRIVATE))
            except ProviderError:
                return
            raise AssertionError('provider failure was not preserved')
        child = multiprocessing.get_context('fork').Process(target=invoke_in_child)
        child.start(); child.join(3)
        hung = child.is_alive()
        if hung:
            child.terminate(); child.join(3)
        self.assertFalse(hung, 'non-regular diagnostic file blocked accounting')
        self.assertEqual(child.exitcode, 0)
        self.assertTrue(stat.S_ISFIFO(self.diagnostic.lstat().st_mode))
        self.uncertain()
