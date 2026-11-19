"""Fresh local permissions, synthetic accounting only; no operational provisioning."""
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import fcntl
import json
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from checkout_fixture import checkout
from budget_fixture import document, Provider, request
from northstar_atlas import local_test_authority as local
from northstar_atlas.budget_transition import _encode
from northstar_atlas.diagnostic_window import sha
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, LINEAR


class FreshAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        (self.root/'.local').mkdir(mode=0o700)
        self.enterContext(patch.dict(os.environ, {'NORTHSTAR_PROJECT_ROOT':str(self.root)}, clear=True))
        self.spec = WorkflowRequest('synthetic-once', LINEAR, 'Who approves access?',
                                    'approved_only', 'chroma', date(2026, 10, 29))
        self.price = document()['pricing_revisions'][0]
        from northstar_atlas import live_access, local_live
        os.environ.update(NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_UI_ENABLE_LIVE='1',
                          VECTOR_STORE='chroma', NORTHSTAR_UI_ENABLE_PINECONE='0')
        self.enterContext(patch('platform.system', return_value='Darwin'))
        self.enterContext(live_access.local_session(self.root, backend='chroma', port=8501,
                                                   enable_graph=False, approved=True))
        options = local_live.server_options(8501)
        from streamlit.config import get_option
        self.enterContext(patch('streamlit.config.get_option', side_effect=lambda key:
            options[key.replace('.', '_')] if key.replace('.', '_') in options else get_option(key)))
        self.enterContext(patch('streamlit.runtime.context._get_client_context', return_value=SimpleNamespace(
            remote_ip='127.0.0.1', headers=[('Host','127.0.0.1:8501'), ('Origin','http://127.0.0.1:8501')])) )

    def prepare(self, **changes):
        options = dict(verified_pricing_sha256=sha(_encode(self.price)),
                       scope_review_sha256='a'*64, approve_new_scope=True)
        options.update(changes)
        self.preview = local.prepare(self.root, self.spec, self.price, **options)
        from northstar_atlas.live_access import require_local_access
        self.budget = PersistentBudget(self.root, scope=self.preview.scope,
            billing_scope=self.preview.billing_scope, policy=BudgetPolicy(1, Decimal('0.010000')),
            before_request=require_local_access)
        return self.preview

    def apply(self, **changes):
        options = dict(approved_plan_sha256=self.preview.plan_sha256,
                       approved_candidate_sha256=self.preview.candidate_sha256,
                       writers_stopped=True, approve_activation=True)
        options.update(changes)
        local.apply(self.budget, **options)
        os.environ.update(NORTHSTAR_LOCAL_TEST_PLAN=self.preview.plan_sha256,
                          NORTHSTAR_LOCAL_TEST_RUN_APPROVAL=local.run_approval(self.preview.plan_sha256))

    def test_prepared_authority_is_inert_until_separate_exact_approval(self):
        self.prepare()
        with self.assertRaises(BudgetError):
            self.budget.check_ready()
        with self.assertRaises(BudgetError):
            self.apply(approve_activation=False)
        self.apply()
        with self.assertRaises(BudgetError):
            self.budget.check_ready()  # Never falls back to unrestricted requests.
        with self.budget.reserve(self.spec, diagnostic_approval=self.preview.plan_sha256) as permit:
            permit.invoke(Provider(), 'generator', 0, request())
        ledger, _, state = self.budget._read()
        self.assertEqual((state.attempts_consumed, ledger['max_attempts']), (1, 1))
        self.assertEqual(ledger['max_reserved_usd'], '0.010000')
        self.assertEqual(ledger['pricing_revisions'], [self.price])
        self.assertIsNone(ledger['legacy'])
        self.assertFalse(self.budget.pending_path.exists())

    def test_existing_authority_is_never_read_or_overwritten(self):
        directory = self.root/'.local/controlled-live'; directory.mkdir()
        private = directory/'budget.json'; private.write_bytes(b'opaque historical sentinel')
        with self.assertRaises((BudgetError, FileExistsError)):
            self.prepare()
        self.assertEqual(private.read_bytes(), b'opaque historical sentinel')

    def test_missing_approval_bad_price_or_other_profile_creates_nothing(self):
        for changes in ({'approve_new_scope':False}, {'verified_pricing_sha256':'0'*64},
                        {'scope_review_sha256':'invalid'}):
            with self.subTest(changes=changes), self.assertRaises(BudgetError): self.prepare(**changes)
            self.assertFalse((self.root/'.local/controlled-live').exists())
        self.spec = replace(self.spec, backend='pinecone')
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse((self.root/'.local/controlled-live').exists())

    def test_tampered_candidate_or_digest_cannot_activate(self):
        self.prepare()
        with self.assertRaises(BudgetError): self.apply(approved_candidate_sha256='0'*64)
        self.budget.path.write_bytes(self.budget.path.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.apply()
        self.assertTrue(self.budget.pending_path.exists())

    def test_activation_expiry_contention_and_identity_change_fail_closed(self):
        self.prepare()
        with self.budget.lock_path.open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BudgetError): self.apply()
        with patch.object(self.budget, 'clock', return_value=self.budget.clock()+timedelta(days=2)):
            with self.assertRaises(BudgetError): self.apply()
        old = self.budget.lock_path.with_name('old-lock'); self.budget.lock_path.rename(old)
        self.budget.lock_path.write_bytes(b''); self.budget.lock_path.chmod(0o600)
        with self.assertRaises(BudgetError): self.apply()
        self.assertTrue(self.budget.pending_path.exists())

    def test_failed_activation_persistence_remains_blocked_and_is_not_retried(self):
        self.prepare()
        with patch.object(self.budget, '_sync_directory', side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(BudgetError): self.apply()
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.apply()
        with self.assertRaises(BudgetError): self.budget.check_ready()

    def test_post_unlink_sync_failure_restores_guard_and_blocks_restart(self):
        self.prepare()
        sync = self.budget._sync_directory
        calls = []
        def fail_final_sync():
            calls.append(True)
            if len(calls) == 2: raise OSError('synthetic final sync failure')
            sync()
        with patch.object(self.budget, '_sync_directory', side_effect=fail_final_sync):
            with self.assertRaises(BudgetError): self.apply()
        self.assertEqual(self.budget.pending_path.read_bytes(), local.GUARD)
        restarted = PersistentBudget(self.root, scope=self.preview.scope,
            billing_scope=self.preview.billing_scope, policy=self.budget.policy)
        with self.assertRaises(BudgetError):
            restarted.inspect_history(diagnostic_approval=self.preview.plan_sha256)

    def test_duplicate_new_id_wrong_question_and_restart_cannot_reopen_allowance(self):
        self.prepare(); self.apply()
        for spec in (replace(self.spec, question='Other?'), replace(self.spec, invocation_id='other'),
                     replace(self.spec, mode='draft_review'), replace(self.spec, backend='pinecone')):
            with self.subTest(field=spec), self.assertRaises(BudgetError):
                with self.budget.reserve(spec, diagnostic_approval=self.preview.plan_sha256): pass
        with self.budget.reserve(self.spec, diagnostic_approval=self.preview.plan_sha256):
            pass  # Even a no-inference abstention consumes the one-shot invocation.
        restarted = PersistentBudget(self.root, scope=self.preview.scope,
            billing_scope=self.preview.billing_scope, policy=self.budget.policy)
        for budget in (self.budget, restarted):
            with self.assertRaises(BudgetError):
                with budget.reserve(self.spec, diagnostic_approval=self.preview.plan_sha256): pass
        with self.assertRaises(BudgetError): self.apply()

    def test_timeout_retains_entire_reservation_and_does_not_retry(self):
        self.prepare(); self.apply()
        provider = Provider(TimeoutError('synthetic'))
        with self.assertRaises(TimeoutError):
            with self.budget.reserve(self.spec, diagnostic_approval=self.preview.plan_sha256) as permit:
                permit.invoke(provider, 'generator', 0, request())
        ledger, _, state = self.budget._read()
        self.assertEqual(ledger['workflows'][0]['status'], 'uncertain')
        self.assertEqual(state.attempts_consumed, 1)
        self.assertEqual(state.reserved_usd, Decimal('0.003687'))
        self.assertEqual(len(ledger['workflows'][0]['calls']), 1)

    def test_pricing_ceiling_expiry_and_ui_question_limit_reject_before_creation(self):
        original = self.price.copy()
        self.price['input_per_million_usd'] = '1'
        with self.assertRaises(BudgetError): self.prepare()
        self.price = dict(original, valid_until=original['confirmed_at'])
        with self.assertRaises(BudgetError): self.prepare()
        self.price = original
        self.spec = replace(self.spec, question='a'*2001)
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse((self.root/'.local/controlled-live').exists())

    def test_missing_window_permissions_and_conflicting_modes_deny(self):
        self.prepare(); self.apply()
        for settings in ({'NORTHSTAR_DEMO_BATCH_ENABLED':'1'}, {'NORTHSTAR_AUTHORITY_HANDOFF_PLAN':'b'*64},
                         {'NORTHSTAR_UI_ENABLE_GRAPH':'1'}, {'NORTHSTAR_LOCAL_TEST_RUN_APPROVAL':''}):
            with patch.dict(os.environ, settings), self.assertRaises(BudgetError):
                local.inspect_invocation(self.budget, self.preview.plan_sha256)
        window = self.budget.directory/'diagnostic-window.json'
        window.chmod(0o644)
        with self.assertRaises(BudgetError): local.inspect_invocation(self.budget, self.preview.plan_sha256)
        window.chmod(0o600); window.rename(window.with_name('preserved-window'))
        with self.assertRaises(BudgetError):
            with self.budget.reserve(self.spec): pass
        self.assertEqual(self.budget._read()[0]['workflows'], [])

    def test_partial_result_claim_spends_invocation_even_without_inference(self):
        self.prepare(); self.apply()
        from northstar_atlas.result_capture import result_path
        claim = result_path(self.root, self.spec.invocation_id)
        claim.write_bytes(b''); claim.chmod(0o600)
        status = local.inspect_invocation(self.budget, self.preview.plan_sha256)
        self.assertFalse(status.available)
        self.assertEqual(self.budget._read()[0]['workflows'], [])

    def test_lower_cost_configuration_must_match_approved_policy(self):
        self.prepare(); self.apply()
        from northstar_atlas.live_budget import configured_budget
        os.environ.update(NORTHSTAR_BUDGET_SCOPE=self.preview.scope,
                          NORTHSTAR_BILLING_SCOPE=self.preview.billing_scope, NORTHSTAR_BUDGET_MAX_ATTEMPTS='1')
        with self.assertRaises(BudgetError): local.inspect_invocation(configured_budget(self.root), self.preview.plan_sha256)
        os.environ['NORTHSTAR_BUDGET_MAX_RESERVED_USD'] = '0.010000'
        self.assertTrue(local.inspect_invocation(configured_budget(self.root), self.preview.plan_sha256).available)
        for invalid in ('nan', '-1', '0.11', '0'):
            with patch.dict(os.environ, NORTHSTAR_BUDGET_MAX_RESERVED_USD=invalid), self.assertRaises(ValueError):
                configured_budget(self.root)


if __name__ == '__main__': unittest.main()
