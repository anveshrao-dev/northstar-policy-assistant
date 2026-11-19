"""Activated local pricing renewal with synthetic accounting and fake providers."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import importlib
import importlib.util
import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import test_local_test_pricing as inert_tests
from northstar_atlas import local_test_authority as local
from northstar_atlas.budget_transition import _decode, _encode
from northstar_atlas.diagnostic_window import sha
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, GRAPH
from budget_fixture import Provider, request


class ActivePricingTests(unittest.TestCase):
    prepare_successor = inert_tests.InertPricingTests.prepare_successor
    apply_successor = inert_tests.InertPricingTests.apply_successor
    activate = inert_tests.InertPricingTests.activate
    restarted = inert_tests.InertPricingTests.restarted

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.local_active_pricing'),
                             'Activated unused authorities need a separate pricing successor')
        self.active = importlib.import_module('northstar_atlas.local_active_pricing')
        inert_tests.InertPricingTests.setUp(self)
        self.prepare_successor(); self.apply_successor(); self.activate()
        self.parent = self.successor
        self.original_active = {str(p.relative_to(self.budget.directory)): p.read_bytes()
            for p in self.budget.directory.rglob('*') if p.is_file()}
        self.now += timedelta(hours=2)
        self.price = dict(self.price, id='synthetic-active-price',
            confirmed_at=(self.now-timedelta(minutes=1)).isoformat(),
            valid_until=(self.now+timedelta(hours=1)).isoformat())

    def prepare_renewal(self, **changes):
        options = dict(previous_activation_sha256=self.parent.plan_sha256,
            verified_pricing_sha256=sha(_encode(self.price)), writers_stopped=True,
            approve_preparation=True)
        options.update(changes)
        self.successor = self.active.prepare(self.budget, self.price, **options)
        return self.successor

    def apply_renewal(self, **changes):
        options = dict(approved_plan_sha256=self.successor.plan_sha256,
            approved_candidate_sha256=self.successor.candidate_sha256,
            writers_stopped=True, approve_application=True)
        options.update(changes)
        self.active.apply(self.budget, **options)

    def select(self):
        os.environ.update(NORTHSTAR_LOCAL_TEST_PLAN=self.successor.plan_sha256,
            NORTHSTAR_LOCAL_TEST_RUN_APPROVAL=local.run_approval(self.successor.plan_sha256))

    def denied(self, approval=None):
        with self.assertRaises(BudgetError):
            with self.restarted().reserve(self.spec,
                    diagnostic_approval=approval or self.successor.plan_sha256):
                self.fail('Denied permission reached resource setup')

    def test_preserves_all_prior_artifacts_and_only_appends_current_price(self):
        self.denied(self.parent.plan_sha256)
        self.prepare_renewal()
        self.assertEqual(self.budget.path.read_bytes(), self.original_active['budget.json'])
        self.apply_renewal(); self.select()
        for name, raw in self.original_active.items():
            if name != 'budget.json': self.assertEqual((self.budget.directory/name).read_bytes(), raw)
        value, _, state = self.budget._read()
        old = _decode(self.original_active['budget.json'])
        self.assertEqual(value['pricing_revisions'][:-1], old['pricing_revisions'])
        self.assertEqual(value['pricing_revisions'][-1], self.price)
        self.assertEqual(value['active_pricing'], 'synthetic-active-price')
        for key in old.keys()-{'pricing_revisions', 'active_pricing'}:
            self.assertEqual(value[key], old[key])
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (0, Decimal(0)))
        self.assertEqual((value['max_attempts'], value['max_reserved_usd']), (1, '0.010000'))
        self.assertTrue(local.inspect_invocation(self.restarted(), self.successor.plan_sha256).available)

    def test_one_fake_call_settles_and_cannot_be_repeated_after_restart(self):
        self.prepare_renewal(); self.apply_renewal(); self.select()
        provider = Provider()
        with self.budget.reserve(self.spec, diagnostic_approval=self.successor.plan_sha256) as permit:
            permit.invoke(provider, 'generator', 0, request())
        value, _, state = self.budget._read()
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (1, Decimal('0.003687')))
        self.assertEqual((value['workflows'][0]['status'], value['workflows'][0]['price_id']),
                         ('closed', 'synthetic-active-price'))
        self.assertFalse(local.inspect_invocation(self.restarted(), self.successor.plan_sha256).available)
        self.denied()
        self.spec = replace(self.spec, invocation_id='different-id'); self.denied()
        self.assertEqual(len(provider.calls), 1)

    def test_timeout_remains_uncertain_without_refund_or_retry(self):
        self.prepare_renewal(); self.apply_renewal(); self.select()
        provider = Provider(TimeoutError('synthetic timeout'))
        with self.assertRaises(TimeoutError):
            with self.budget.reserve(self.spec, diagnostic_approval=self.successor.plan_sha256) as permit:
                permit.invoke(provider, 'generator', 0, request())
        value, _, state = self.budget._read()
        self.assertEqual(value['workflows'][0]['status'], 'uncertain')
        self.assertEqual((state.attempts_consumed, state.reserved_usd), (1, Decimal('0.003687')))
        self.denied(); self.assertEqual(len(provider.calls), 1)

    def test_preparation_and_application_require_separate_exact_approvals(self):
        for change in ({'writers_stopped':False}, {'approve_preparation':False},
                       {'previous_activation_sha256':'0'*64}, {'verified_pricing_sha256':'0'*64}):
            with self.subTest(change=change), self.assertRaises(BudgetError): self.prepare_renewal(**change)
        self.prepare_renewal()
        for change in ({'writers_stopped':False}, {'approve_application':False},
                       {'approved_plan_sha256':'0'*64}, {'approved_candidate_sha256':'0'*64}):
            with self.subTest(change=change), self.assertRaises(BudgetError): self.apply_renewal(**change)
        self.assertEqual(self.budget.path.read_bytes(), self.original_active['budget.json'])

    def test_invalid_expired_future_reused_or_over_cap_prices_reject(self):
        for changes in ({'valid_until':(self.now-timedelta(seconds=1)).isoformat()},
                        {'confirmed_at':(self.now+timedelta(minutes=1)).isoformat()},
                        {'valid_until':(self.now+timedelta(days=2)).isoformat()},
                        {'id':'synthetic-successor'}, {'source':'https://example.invalid/'},
                        {'input_per_million_usd':'1'}, {'output_per_million_usd':'NaN'}):
            with self.subTest(changes=changes):
                price = self.price; self.price = dict(price, **changes)
                with self.assertRaises(BudgetError): self.prepare_renewal()
                self.price = price
        self.assertFalse((self.budget.directory/local.ACTIVE_PRICING_DIRECTORY).exists())

    def test_unexpired_parent_cannot_renew(self):
        self.now -= timedelta(hours=2)
        with self.assertRaises(BudgetError): self.prepare_renewal()

    def test_original_unrenewed_activation_can_use_same_narrow_successor(self):
        # A second synthetic checkout exercises the original version-1 approval.
        import test_local_test_authority as fresh_tests
        from northstar_atlas import live_access
        # Close the first fixture's local bootstrap before selecting the second.
        live_access._local = None
        fresh_tests.FreshAuthorityTests.setUp(self)
        fresh_tests.FreshAuthorityTests.prepare(self)
        fresh_tests.FreshAuthorityTests.apply(self)
        self.parent = self.preview
        self.original_active = {str(p.relative_to(self.budget.directory)):p.read_bytes()
            for p in self.budget.directory.rglob('*') if p.is_file()}
        self.now = self.budget.clock()+timedelta(hours=2)
        self.budget.clock=lambda:self.now
        self.price=dict(self.price,id='active-from-original',
            confirmed_at=(self.now-timedelta(minutes=1)).isoformat(),
            valid_until=(self.now+timedelta(hours=1)).isoformat())
        self.prepare_renewal();self.apply_renewal();self.select()
        self.assertTrue(local.inspect_invocation(self.budget,self.successor.plan_sha256).available)
        self.assertEqual((self.budget.directory/'diagnostic-window.json').read_bytes(),
                         self.original_active['diagnostic-window.json'])

    def test_duplicate_prepare_and_apply_are_rejected(self):
        self.prepare_renewal()
        with self.assertRaises(BudgetError): self.prepare_renewal()
        self.apply_renewal(); raw=self.budget.path.read_bytes()
        with self.assertRaises(BudgetError): self.apply_renewal()
        self.assertEqual(self.budget.path.read_bytes(), raw)

    def test_old_approval_is_fenced_at_prepare_and_after_application(self):
        self.prepare_renewal()
        self.now -= timedelta(hours=2)  # Even when the historical price would be current.
        self.denied(self.parent.plan_sha256)
        self.now += timedelta(hours=2)
        self.apply_renewal()
        self.denied(self.parent.plan_sha256)
        os.environ['NORTHSTAR_LOCAL_TEST_PLAN'] = self.successor.plan_sha256
        self.denied()  # Old run approval must not silently authorize successor.

    def test_mismatched_request_cannot_reserve_after_renewal(self):
        self.prepare_renewal(); self.apply_renewal(); self.select()
        original = self.spec
        for change in ({'question':'Different?'}, {'mode':'draft_review'}, {'backend':'pinecone'},
                       {'reference_date':original.reference_date+timedelta(days=1)}, {'profile':GRAPH}):
            self.spec = replace(original, **change); self.denied()
        self.assertEqual(self.budget._read()[0]['workflows'], [])

    def test_consumed_or_uncertain_parent_cannot_renew(self):
        self.now -= timedelta(hours=2)
        with self.budget.reserve(self.spec, diagnostic_approval=self.parent.plan_sha256): pass
        self.now += timedelta(hours=2)
        with self.assertRaises(BudgetError): self.prepare_renewal()

    def test_existing_result_claim_or_guard_cannot_renew(self):
        from northstar_atlas.result_capture import result_path
        claim = result_path(self.root, self.spec.invocation_id)
        claim.write_bytes(b''); claim.chmod(0o600)
        with self.assertRaises(BudgetError): self.prepare_renewal()
        claim.unlink()
        self.budget.pending_path.write_bytes(b'uncertain'); self.budget.pending_path.chmod(0o600)
        with self.assertRaises(BudgetError): self.prepare_renewal()

    def test_missing_activation_cannot_renew(self):
        (self.budget.directory/'diagnostic-window.json').unlink()
        with self.assertRaises(BudgetError): self.prepare_renewal()

    def test_changed_source_or_activation_bytes_invalidate_application(self):
        self.prepare_renewal()
        for path in (self.budget.path, self.budget.directory/'diagnostic-window.json'):
            raw=path.read_bytes(); path.write_bytes(raw+b' ')
            with self.assertRaises(BudgetError): self.apply_renewal()
            path.write_bytes(raw)
        self.assertFalse(self.budget.pending_path.exists())

    def test_replaced_ledger_inode_invalidates_candidate_even_with_same_bytes(self):
        self.prepare_renewal()
        p=self.budget.path; raw=p.read_bytes();p.rename(p.with_name('synthetic-old-ledger'))
        p.write_bytes(raw);p.chmod(0o600)
        with self.assertRaises(BudgetError): self.apply_renewal()

    def test_changed_lock_identity_rejects_application(self):
        self.prepare_renewal()
        p=self.budget.lock_path;p.rename(p.with_name('synthetic-old-lock'));p.write_bytes(b'');p.chmod(0o600)
        with self.assertRaises(BudgetError): self.apply_renewal()

    def test_tampered_candidate_and_rehashed_request_reject(self):
        self.prepare_renewal();directory=self.budget.directory/local.ACTIVE_PRICING_DIRECTORY
        p=directory/self.active.PLAN;raw=p.read_bytes();plan=_decode(raw)
        plan['request']['question_sha256']='b'*64;p.write_bytes(_encode(plan))
        with self.assertRaises(BudgetError): self.apply_renewal(approved_plan_sha256=sha(_encode(plan)))
        p.write_bytes(raw)
        p=directory/self.active.CANDIDATE;value=_decode(p.read_bytes())
        value['pricing_revisions'][0]['input_per_million_usd']='0.09';p.write_bytes(_encode(value))
        with self.assertRaises(BudgetError): self.apply_renewal()

    def test_scope_billing_and_limits_cannot_change(self):
        self.prepare_renewal()
        for change in ({'scope':'f'*32},{'billing_scope':'other'},
                       {'policy':BudgetPolicy(2,Decimal('.01'))},{'policy':BudgetPolicy(1,Decimal('.1'))}):
            options=dict(scope=self.budget.scope,billing_scope=self.budget.billing_scope,
                         policy=local.POLICY,clock=self.budget.clock);options.update(change)
            b=PersistentBudget(self.root,**options)
            with self.assertRaises(BudgetError):
                self.active.apply(b,approved_plan_sha256=self.successor.plan_sha256,
                    approved_candidate_sha256=self.successor.candidate_sha256,
                    writers_stopped=True,approve_application=True)

    def test_process_lock_contention_prevents_prepare_and_apply(self):
        script='import fcntl,sys;f=open(sys.argv[1],"r+");fcntl.flock(f,fcntl.LOCK_EX);print("locked",flush=True);sys.stdin.readline()'
        for prepared in (False,True):
            if prepared:self.prepare_renewal()
            process=subprocess.Popen([sys.executable,'-I','-B','-c',script,str(self.budget.lock_path)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(),'locked')
                with self.assertRaises(BudgetError):
                    self.apply_renewal() if prepared else self.prepare_renewal()
            finally:process.communicate('\n',timeout=10)

    def test_partial_preparation_blocks_old_approval_after_restart(self):
        with patch.object(self.active,'_write_private',side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(BudgetError):self.prepare_renewal()
        self.now-=timedelta(hours=2);self.denied(self.parent.plan_sha256)

    def test_atomic_write_failure_leaves_guards_and_rejects_retry(self):
        self.prepare_renewal()
        with patch.object(self.budget,'_write',side_effect=OSError('synthetic replacement failure')):
            with self.assertRaises(BudgetError):self.apply_renewal()
        self.assertTrue(self.budget.pending_path.exists());self.select();self.denied()
        self.budget=self.restarted()
        with self.assertRaises(BudgetError):self.apply_renewal()
        self.assertEqual(self.budget.path.read_bytes(),self.original_active['budget.json'])

    def test_interruption_after_ledger_replace_blocks_restart(self):
        self.prepare_renewal();write=self.budget._write
        def interrupt(value):write(value);raise KeyboardInterrupt('synthetic interruption')
        with patch.object(self.budget,'_write',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):self.apply_renewal()
        self.select();self.denied();self.assertTrue(self.budget.pending_path.exists())

    def test_receipt_failure_keeps_global_fence(self):
        self.prepare_renewal();write=self.active._write_private
        def fail(b,p,raw):
            if p.name==self.active.APPLIED:raise OSError('synthetic receipt failure')
            return write(b,p,raw)
        with patch.object(self.active,'_write_private',side_effect=fail):
            with self.assertRaises(BudgetError):self.apply_renewal()
        self.select();self.denied();self.assertTrue(self.budget.pending_path.exists())

    def test_final_fsync_failure_restores_execution_fence(self):
        self.prepare_renewal();sync=self.budget._sync_directory
        def fail():
            if not self.budget.pending_path.exists():raise OSError('synthetic final sync failure')
            return sync()
        with patch.object(self.budget,'_sync_directory',side_effect=fail):
            with self.assertRaises(BudgetError):self.apply_renewal()
        self.select();self.denied();self.assertTrue(self.budget.pending_path.exists())

    def test_interrupt_after_final_guard_unlink_restores_execution_fence(self):
        self.prepare_renewal();sync=self.budget._sync_directory
        def interrupt():
            if not self.budget.pending_path.exists():raise KeyboardInterrupt('synthetic final interruption')
            return sync()
        with patch.object(self.budget,'_sync_directory',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):self.apply_renewal()
        self.assertTrue(self.budget.pending_path.exists())
        self.select();self.denied()

    def test_new_expiry_blocks_application_and_runtime(self):
        self.prepare_renewal();self.now+=timedelta(hours=2)
        with self.assertRaises(BudgetError):self.apply_renewal()
        self.now-=timedelta(hours=2);self.apply_renewal();self.select()
        self.now+=timedelta(hours=2);self.denied()

    def test_malformed_completion_receipt_or_private_permissions_fail_closed(self):
        self.prepare_renewal();self.apply_renewal();self.select()
        directory=self.budget.directory/local.ACTIVE_PRICING_DIRECTORY
        p=directory/self.active.APPLIED;raw=p.read_bytes();p.write_bytes(b'{}');self.denied();p.write_bytes(raw)
        directory.chmod(0o755);self.denied()

    def test_non_object_candidate_is_a_safe_runtime_denial(self):
        self.prepare_renewal();self.apply_renewal();self.select()
        path=self.budget.directory/local.ACTIVE_PRICING_DIRECTORY/self.active.CANDIDATE
        for raw in (b'null', b'[]', b'"invalid"'):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                self.denied()

    def test_cli_uses_same_approved_prepare_apply_contract(self):
        price=self.root/'.local/price.json';price.write_bytes(_encode(self.price));price.chmod(0o600)
        with patch.object(self.active,'PersistentBudget',return_value=self.budget),patch.object(self.active.Path,'cwd',return_value=self.root):
            output=io.StringIO()
            with redirect_stdout(output):
                code=self.active.main(['prepare','--scope',self.budget.scope,'--billing-scope',self.budget.billing_scope,
                    '--pricing-file','.local/price.json','--previous-activation-sha256',self.parent.plan_sha256,
                    '--verified-pricing-sha256',sha(_encode(self.price)),'--writers-stopped','--approve-preparation'])
            self.assertEqual(code,0);result=_decode(output.getvalue().encode())
            with redirect_stdout(io.StringIO()):
                code=self.active.main(['apply','--scope',self.budget.scope,'--billing-scope',self.budget.billing_scope,
                    '--approved-plan-sha256',result['plan_sha256'],'--approved-candidate-sha256',result['candidate_sha256'],
                    '--writers-stopped','--approve-application'])
            self.assertEqual(code,0)
        self.assertEqual((self.budget.directory/'diagnostic-window.json').read_bytes(),self.original_active['diagnostic-window.json'])


if __name__=='__main__':unittest.main()
