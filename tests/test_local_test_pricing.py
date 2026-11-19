"""Synthetic inert pricing successors; never opens operational accounting."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from dataclasses import replace
import importlib
import importlib.util
import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import test_local_test_authority as fresh_tests
from northstar_atlas import local_test_authority as local
from northstar_atlas.budget_transition import _decode, _encode
from northstar_atlas.diagnostic_window import sha
from northstar_atlas.live_budget import BudgetError, PersistentBudget
from budget_fixture import Provider, request


class InertPricingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.local_test_pricing'),
                             'A separate inert pricing successor operation is required')
        self.renew = importlib.import_module('northstar_atlas.local_test_pricing')
        fresh_tests.FreshAuthorityTests.setUp(self)
        fresh_tests.FreshAuthorityTests.prepare(self)
        self.original = {p.name:p.read_bytes() for p in self.budget.directory.iterdir()}
        self.now = datetime.now(timezone.utc)+timedelta(hours=2)
        self.budget.clock = lambda:self.now
        self.price = dict(self.price, id='synthetic-successor',
            confirmed_at=(self.now-timedelta(minutes=1)).isoformat(),
            valid_until=(self.now+timedelta(hours=1)).isoformat())

    def prepare_successor(self, **changes):
        options = dict(previous_plan_sha256=self.preview.plan_sha256,
            verified_pricing_sha256=sha(_encode(self.price)), writers_stopped=True,
            approve_preparation=True)
        options.update(changes)
        self.successor = self.renew.prepare(self.budget, self.price, **options)
        return self.successor

    def apply_successor(self, **changes):
        options = dict(approved_plan_sha256=self.successor.plan_sha256,
            approved_candidate_sha256=self.successor.candidate_sha256,
            writers_stopped=True, approve_application=True)
        options.update(changes)
        self.renew.apply(self.budget, **options)

    def activate(self):
        local.apply(self.budget, approved_plan_sha256=self.successor.plan_sha256,
            approved_candidate_sha256=self.successor.candidate_sha256,
            writers_stopped=True, approve_activation=True)
        os.environ.update(NORTHSTAR_LOCAL_TEST_PLAN=self.successor.plan_sha256,
            NORTHSTAR_LOCAL_TEST_RUN_APPROVAL=local.run_approval(self.successor.plan_sha256))

    def restarted(self):
        return PersistentBudget(self.root, scope=self.budget.scope,
            billing_scope=self.budget.billing_scope, policy=local.POLICY,
            clock=self.budget.clock, before_request=self.budget.before_request)

    def assert_activation_denied(self):
        budget = self.restarted()
        with self.assertRaises(BudgetError):
            local.apply(budget, approved_plan_sha256=self.successor.plan_sha256,
                approved_candidate_sha256=self.successor.candidate_sha256,
                writers_stopped=True, approve_activation=True)
        self.assertEqual(budget.pending_path.read_bytes(), local.GUARD)
        self.assertFalse((budget.directory/'diagnostic-window.json').exists())

    def test_successor_keeps_original_artifacts_and_guard_until_separate_activation(self):
        with self.assertRaises(BudgetError):
            local.validate_plan(self.budget, _decode(self.original[local.PLAN]))
        self.prepare_successor()
        for name, raw in self.original.items():
            self.assertEqual((self.budget.directory/name).read_bytes(), raw)
        self.apply_successor()
        for name in (local.PLAN, local.CANDIDATE, 'budget.lock', 'settlement.pending'):
            self.assertEqual((self.budget.directory/name).read_bytes(), self.original[name])
        value, _, state = self.budget._read()
        original = _decode(self.original['budget.json'])
        self.assertEqual(value['pricing_revisions'][:-1], original['pricing_revisions'])
        self.assertEqual(value['pricing_revisions'][-1], self.price)
        self.assertEqual(value['active_pricing'], 'synthetic-successor')
        for key in original.keys()-{'pricing_revisions','active_pricing'}:
            self.assertEqual(value[key], original[key])
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (0, Decimal('0')))
        with self.assertRaises(BudgetError): self.budget.check_ready()
        with self.assertRaises(BudgetError):
            local.apply(self.budget, approved_plan_sha256=self.successor.plan_sha256,
                approved_candidate_sha256=self.successor.candidate_sha256,
                writers_stopped=True, approve_activation=False)

    def test_new_approval_allows_one_fake_call_and_existing_result_claim(self):
        self.prepare_successor(); self.apply_successor(); self.activate()
        self.assertTrue(local.inspect_invocation(self.budget, self.successor.plan_sha256).available)
        from northstar_atlas.result_capture import result_path
        claim = result_path(self.root, self.spec.invocation_id)
        claim.write_bytes(b''); claim.chmod(0o600)
        provider = Provider()
        with self.budget.reserve(self.spec, diagnostic_approval=self.successor.plan_sha256) as permit:
            permit.invoke(provider, 'generator', 0, request())
        ledger, _, state = self.budget._read()
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (1, Decimal('0.003687')))
        self.assertEqual(ledger['workflows'][0]['price_id'], 'synthetic-successor')
        self.assertFalse(local.inspect_invocation(self.restarted(), self.successor.plan_sha256).available)
        for spec in (self.spec, replace(self.spec, invocation_id='another-id')):
            with self.assertRaises(BudgetError):
                with self.restarted().reserve(spec, diagnostic_approval=self.successor.plan_sha256): pass
        self.assertEqual(len(provider.calls), 1)

    def test_wrong_request_and_old_run_approval_are_rejected(self):
        self.prepare_successor(); self.apply_successor(); self.activate()
        for spec in (replace(self.spec, question='Changed?'), replace(self.spec, mode='draft_review'),
                     replace(self.spec, backend='pinecone'), replace(self.spec, reference_date=self.spec.reference_date+timedelta(days=1))):
            with self.assertRaises(BudgetError):
                with self.budget.reserve(spec, diagnostic_approval=self.successor.plan_sha256): pass
        with self.assertRaises(BudgetError):
            with self.budget.reserve(self.spec, diagnostic_approval=self.preview.plan_sha256): pass
        self.assertEqual(self.budget._read()[0]['workflows'], [])

    def test_fake_timeout_keeps_full_charge_without_retry(self):
        self.prepare_successor(); self.apply_successor(); self.activate()
        provider = Provider(TimeoutError('synthetic'))
        with self.assertRaises(TimeoutError):
            with self.budget.reserve(self.spec, diagnostic_approval=self.successor.plan_sha256) as permit:
                permit.invoke(provider, 'generator', 0, request())
        ledger, _, state = self.budget._read()
        self.assertEqual(ledger['workflows'][0]['status'], 'uncertain')
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (1, Decimal('0.003687')))
        self.assertEqual(len(provider.calls), 1)

    def test_missing_preparation_approval_wrong_parent_or_unreviewed_price_creates_nothing(self):
        for change in ({'approve_preparation':False}, {'writers_stopped':False},
                       {'previous_plan_sha256':'0'*64}, {'verified_pricing_sha256':'0'*64}):
            with self.subTest(change=change), self.assertRaises(BudgetError): self.prepare_successor(**change)
            self.assertFalse((self.budget.directory/local.PRICING_DIRECTORY).exists())
            self.assertEqual(self.budget.path.read_bytes(), self.original['budget.json'])

    def test_price_expiry_future_start_duplicate_id_bad_url_and_cost_reject(self):
        for changes in (
            {'confirmed_at':(self.now-timedelta(hours=2)).isoformat(), 'valid_until':(self.now-timedelta(hours=1)).isoformat()},
            {'confirmed_at':(self.now+timedelta(minutes=1)).isoformat()},
            {'valid_until':(self.now+timedelta(days=2)).isoformat()},
            {'id':'fixture-price'}, {'source':'https://example.invalid/prices'},
            {'input_per_million_usd':'1'}, {'input_per_million_usd':'NaN'}, {'extra':'field'}):
            with self.subTest(changes=changes):
                old=self.price; self.price=dict(old, **changes)
                with self.assertRaises(BudgetError): self.prepare_successor()
                self.price=old
                self.assertFalse((self.budget.directory/local.PRICING_DIRECTORY).exists())

    def test_unexpired_original_cannot_be_superseded(self):
        self.budget.clock=lambda:datetime.now(timezone.utc)
        with self.assertRaises(BudgetError): self.prepare_successor()
        self.assertFalse((self.budget.directory/local.PRICING_DIRECTORY).exists())

    def test_original_ledger_or_guard_change_rejects_preparation(self):
        for name in ('budget.json','settlement.pending',local.PLAN,local.CANDIDATE):
            with self.subTest(name=name):
                path=self.budget.directory/name; raw=path.read_bytes(); path.write_bytes(raw+b' ')
                with self.assertRaises(BudgetError): self.prepare_successor()
                path.write_bytes(raw)
        self.assertFalse((self.budget.directory/local.PRICING_DIRECTORY).exists())

    def test_prior_activation_retirement_and_result_claim_reject(self):
        for name in ('diagnostic-window.json','authority-retired.json'):
            path=self.budget.directory/name; path.write_bytes(b'partial'); path.chmod(0o600)
            with self.assertRaises(BudgetError): self.prepare_successor()
            path.unlink()
        from northstar_atlas.result_capture import result_path
        claim=result_path(self.root,self.spec.invocation_id); claim.write_bytes(b''); claim.chmod(0o600)
        with self.assertRaises(BudgetError): self.prepare_successor()

    def test_duplicate_preparation_and_application_never_reset(self):
        self.prepare_successor()
        with self.assertRaises(BudgetError): self.prepare_successor()
        self.apply_successor(); raw=self.budget.path.read_bytes()
        with self.assertRaises(BudgetError): self.apply_successor()
        self.assertEqual(self.budget.path.read_bytes(), raw)
        self.assertEqual(self.budget.pending_path.read_bytes(), local.GUARD)

    def test_wrong_application_approval_leaves_original_and_cannot_activate(self):
        self.prepare_successor()
        for change in ({'approve_application':False},{'writers_stopped':False},
                       {'approved_plan_sha256':'0'*64},{'approved_candidate_sha256':'0'*64}):
            with self.subTest(change=change), self.assertRaises(BudgetError): self.apply_successor(**change)
            self.assertEqual(self.budget.path.read_bytes(), self.original['budget.json'])
            self.assert_activation_denied()

    def test_candidate_request_and_price_tampering_is_not_approved_by_rehashing(self):
        self.prepare_successor(); directory=self.budget.directory/local.PRICING_DIRECTORY
        path=directory/self.renew.PLAN; raw=path.read_bytes()
        plan=_decode(raw); plan['request']['question_sha256']='b'*64
        path.write_bytes(_encode(plan))
        with self.assertRaises(BudgetError):
            self.apply_successor(approved_plan_sha256=sha(_encode(plan)))
        path.write_bytes(raw)
        candidate=directory/self.renew.CANDIDATE; value=_decode(candidate.read_bytes())
        value['pricing_revisions'][0]['input_per_million_usd']='0.09'
        candidate.write_bytes(_encode(value))
        with self.assertRaises(BudgetError): self.apply_successor()
        self.assert_activation_denied()

    def test_original_changes_after_preparation_invalidate_candidate(self):
        self.prepare_successor()
        self.budget.path.write_bytes(self.original['budget.json']+b' ')
        with self.assertRaises(BudgetError): self.apply_successor()
        self.assert_activation_denied()

    def test_old_approval_and_clock_rollback_cannot_bypass_successor_fence(self):
        self.prepare_successor()
        self.budget.clock=lambda:datetime.now(timezone.utc)
        with self.assertRaises(BudgetError):
            local.apply(self.budget, approved_plan_sha256=self.preview.plan_sha256,
                approved_candidate_sha256=self.preview.candidate_sha256,
                writers_stopped=True, approve_activation=True)

    def test_lock_contention_between_processes_preserves_original(self):
        script='import fcntl,sys; f=open(sys.argv[1],"r+"); fcntl.flock(f,fcntl.LOCK_EX); print("locked",flush=True); sys.stdin.readline()'
        process=subprocess.Popen([sys.executable,'-I','-B','-c',script,str(self.budget.lock_path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(),'locked')
            with self.assertRaises(BudgetError): self.prepare_successor()
        finally:
            process.communicate('\n',timeout=10)
        self.assertEqual(self.budget.path.read_bytes(),self.original['budget.json'])
        self.assertFalse((self.budget.directory/local.PRICING_DIRECTORY).exists())

    def test_changed_lock_identity_and_unsafe_successor_permissions_reject(self):
        self.prepare_successor()
        directory=self.budget.directory/local.PRICING_DIRECTORY; directory.chmod(0o755)
        with self.assertRaises(BudgetError): self.apply_successor()
        directory.chmod(0o700)
        self.budget.lock_path.rename(self.budget.lock_path.with_name('old-lock'))
        self.budget.lock_path.write_bytes(b''); self.budget.lock_path.chmod(0o600)
        with self.assertRaises(BudgetError): self.apply_successor()
        self.assert_activation_denied()

    def test_preparation_write_failure_fences_original_activation(self):
        with patch.object(self.renew,'_write_private',side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(BudgetError): self.prepare_successor()
        self.assertTrue((self.budget.directory/local.PRICING_DIRECTORY).exists())
        self.assertEqual(self.budget.path.read_bytes(),self.original['budget.json'])
        self.budget.clock=lambda:datetime.now(timezone.utc)
        with self.assertRaises((OSError,ValueError)):
            local.apply(self.budget,approved_plan_sha256=self.preview.plan_sha256,
                approved_candidate_sha256=self.preview.candidate_sha256,writers_stopped=True,approve_activation=True)

    def test_atomic_write_failure_blocks_retry_and_activation_after_restart(self):
        self.prepare_successor()
        with patch.object(self.budget,'_write',side_effect=OSError('synthetic replacement failure')):
            with self.assertRaises(BudgetError): self.apply_successor()
        self.assertEqual(self.budget.path.read_bytes(),self.original['budget.json'])
        self.assert_activation_denied()
        with self.assertRaises(BudgetError):
            self.renew.apply(self.restarted(),approved_plan_sha256=self.successor.plan_sha256,
                approved_candidate_sha256=self.successor.candidate_sha256,writers_stopped=True,approve_application=True)

    def test_failure_after_replacement_preserves_guard_and_original_seed(self):
        self.prepare_successor()
        with patch.object(self.budget,'_sync_directory',side_effect=OSError('synthetic fsync failure')):
            with self.assertRaises(BudgetError): self.apply_successor()
        self.assertEqual((self.budget.directory/local.CANDIDATE).read_bytes(),self.original[local.CANDIDATE])
        self.assert_activation_denied()

    def test_final_pending_unlink_sync_failure_restores_activation_block(self):
        self.prepare_successor(); sync=self.renew._sync
        directory=self.budget.directory/local.PRICING_DIRECTORY
        def fail_final(path):
            if (directory/self.renew.APPLIED).exists() and not (directory/self.renew.PENDING).exists():
                raise OSError('synthetic final sync failure')
            return sync(path)
        with patch.object(self.renew,'_sync',side_effect=fail_final):
            with self.assertRaises(BudgetError): self.apply_successor()
        self.assertTrue((directory/self.renew.PENDING).exists())
        self.assert_activation_denied()

    def test_new_pricing_expiry_rejects_application_and_activation(self):
        self.prepare_successor(); self.now+=timedelta(hours=2)
        with self.assertRaises(BudgetError): self.apply_successor()
        self.assert_activation_denied()

    def test_applied_successor_expiry_does_not_allow_activation_or_another_successor(self):
        self.prepare_successor(); self.apply_successor(); raw=self.budget.path.read_bytes()
        self.now+=timedelta(hours=2)
        self.assert_activation_denied()
        with self.assertRaises(BudgetError): self.prepare_successor()
        self.assertEqual(self.budget.path.read_bytes(),raw)

    def test_forged_or_partial_application_receipt_cannot_activate(self):
        self.prepare_successor(); self.apply_successor()
        path=self.budget.directory/local.PRICING_DIRECTORY/self.renew.APPLIED
        raw=path.read_bytes()
        for replacement in (b'',b'{}',raw+b' '):
            path.write_bytes(replacement)
            self.assert_activation_denied()
        path.write_bytes(raw)

    def test_other_scope_billing_and_limit_cannot_apply_successor(self):
        from northstar_atlas.live_budget import BudgetPolicy
        self.prepare_successor()
        for options in ({'scope':'f'*32},{'billing_scope':'different-local-scope'},
                        {'policy':BudgetPolicy(2,Decimal('0.01'))},
                        {'policy':BudgetPolicy(1,Decimal('0.1'))}):
            values=dict(scope=self.budget.scope,billing_scope=self.budget.billing_scope,
                        policy=local.POLICY,clock=self.budget.clock)
            values.update(options); budget=PersistentBudget(self.root,**values)
            with self.assertRaises(BudgetError):
                self.renew.apply(budget,approved_plan_sha256=self.successor.plan_sha256,
                    approved_candidate_sha256=self.successor.candidate_sha256,
                    writers_stopped=True,approve_application=True)
        self.assertEqual(self.budget.path.read_bytes(),self.original['budget.json'])

    def test_receipt_write_failure_after_replacement_blocks_activation(self):
        self.prepare_successor(); write=self.renew._write_private
        def fail_receipt(budget,path,raw):
            if path.name==self.renew.APPLIED: raise OSError('synthetic receipt failure')
            return write(budget,path,raw)
        with patch.object(self.renew,'_write_private',side_effect=fail_receipt):
            with self.assertRaises(BudgetError): self.apply_successor()
        self.assert_activation_denied()

    def test_expired_initial_and_successor_operations_never_contact_a_provider(self):
        import socket
        with patch.object(socket.socket,'connect',side_effect=AssertionError('provider I/O prohibited')):
            self.prepare_successor(); self.apply_successor()
            self.assertEqual(self.budget._read()[0]['workflows'],[])
            self.assertEqual(self.budget.pending_path.read_bytes(),local.GUARD)

    def test_cli_prepare_apply_and_separate_activation(self):
        pricing=self.root/'.local/review-price.json'; pricing.write_bytes(_encode(self.price));pricing.chmod(0o600)
        # Use the actual CLI against only the synthetic checkout and clock.
        with patch.object(self.renew,'PersistentBudget',return_value=self.budget), patch.object(self.renew.Path,'cwd',return_value=self.root):
            output=io.StringIO()
            with redirect_stdout(output):
                code=self.renew.main(['prepare','--scope',self.budget.scope,'--billing-scope',self.budget.billing_scope,
                    '--pricing-file','.local/review-price.json','--previous-plan-sha256',self.preview.plan_sha256,
                    '--verified-pricing-sha256',sha(_encode(self.price)),'--writers-stopped','--approve-preparation'])
            self.assertEqual(code,0)
            result=_decode(output.getvalue().encode())
            with redirect_stdout(io.StringIO()):
                code=self.renew.main(['apply','--scope',self.budget.scope,'--billing-scope',self.budget.billing_scope,
                    '--approved-plan-sha256',result['plan_sha256'],'--approved-candidate-sha256',result['candidate_sha256'],
                    '--writers-stopped','--approve-application'])
            self.assertEqual(code,0)
        self.assertEqual(self.budget.pending_path.read_bytes(),local.GUARD)
        self.assertFalse((self.budget.directory/'diagnostic-window.json').exists())


if __name__ == '__main__': unittest.main()
