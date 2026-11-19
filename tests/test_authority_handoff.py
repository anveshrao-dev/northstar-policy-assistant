"""Authority continuity uses disposable synthetic histories, never real ledgers."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from contextlib import contextmanager
import importlib
import json
import multiprocessing
import os
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from budget_fixture import Provider, install, request
from checkout_fixture import checkout
import test_demo_batch_pricing as fixtures
from northstar_atlas.budget_transition import _encode
from northstar_atlas.diagnostic_window import WINDOW_NAME, sha
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget
from northstar_atlas import live_access, local_live


class HandoffTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.RenewalTests('runTest'); f.setUp(); self.addCleanup(f.doCleanups)
        renewed = f.prepare(); f.apply(renewed)
        self.f, self.source, self.previous = f, f.budget, renewed.plan_sha256
        self.original = self.source.path.read_bytes()
        self.window = (self.source.directory/WINDOW_NAME).read_bytes()
        self.destination_root = checkout(self)
        path = install(self.destination_root, json.loads(self.original))
        path.write_bytes(self.original)
        old_window = path.parent/WINDOW_NAME
        old_window.write_bytes(self.window); old_window.chmod(0o600)
        self.destination = PersistentBudget(self.source.root, scope=self.source.scope,
            billing_scope=self.source.billing_scope, policy=BudgetPolicy(12), clock=lambda: f.now,
            authority_root=self.destination_root, authorized_authority_root=self.destination_root)
        self.spec = replace(f.f.spec, invocation_id='approved-local-once',
                            question='What does the synthetic HR policy establish?', mode='approved_only')
        self.evidence = sha(b'synthetic reviewed stopped-writer inventory; not a deployment attestation')
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        os.environ.update(NORTHSTAR_PROJECT_ROOT=str(self.source.root), NORTHSTAR_CONTROLLED_LIVE='1',
            NORTHSTAR_UI_ENABLE_LIVE='1', NORTHSTAR_UI_ENABLE_PINECONE='1',
            NORTHSTAR_UI_ENABLE_GRAPH='0', VECTOR_STORE='pinecone')
        self.enterContext(patch('platform.system', return_value='Darwin'))
        options = local_live.server_options(8501)
        from streamlit.config import get_option
        self.enterContext(patch('streamlit.config.get_option', side_effect=lambda key:
            options[key.replace('.', '_')] if key.replace('.', '_') in options else get_option(key)))
        self.client = SimpleNamespace(remote_ip='127.0.0.1',
            headers=[('Host','127.0.0.1:8501'), ('Origin','http://127.0.0.1:8501')])
        self.enterContext(patch('streamlit.runtime.context._get_client_context', side_effect=lambda: self.client))
        self.enterContext(live_access.local_session(self.source.root, backend='pinecone', port=8501,
                                                   enable_graph=False, approved=True))
        self.destination.before_request = live_access.require_local_access

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.authority_handoff'),
                             'authority continuity is not implemented')
        return importlib.import_module('northstar_atlas.authority_handoff')

    def prepare(self, **changes):
        m = self.module()
        description = m.describe_destination(self.destination)
        options = dict(previous_plan_sha256=self.previous, destination=description,
            approved_destination_sha256=sha(_encode(description)), writers_stopped=True,
            reviewed_writer_evidence_sha256=self.evidence,
            expires_at=(self.f.now+timedelta(minutes=30)).isoformat(), dry_run=False)
        options.update(changes)
        return m.prepare_handoff(self.source, self.spec, **options)

    def retire(self, preview, **changes):
        options = dict(approved_plan_sha256=preview.plan_sha256, writers_stopped=True,
                       reviewed_writer_evidence_sha256=self.evidence)
        options.update(changes)
        return self.module().retire_source(self.source, preview.plan_sha256, **options)

    def transfer(self, preview):
        # Explicit test-only staging. Production code never copies authorities.
        target = self.destination.directory/preview.artifact_directory.name
        shutil.copytree(preview.artifact_directory, target)
        return target

    def activate(self, preview, receipt, **changes):
        options = dict(approved_plan_sha256=preview.plan_sha256,
            approved_retirement_sha256=receipt, writers_stopped=True)
        options.update(changes)
        return self.module().activate_destination(self.destination, preview.plan_sha256, **options)

    def ready(self):
        preview = self.prepare(); receipt = self.retire(preview)
        self.transfer(preview); self.activate(preview, receipt)
        os.environ['NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL'] = self.module().run_approval(preview.plan_sha256)
        return preview, receipt

    def invoke(self, plan, spec=None, failure=None):
        with self.destination.reserve(spec or self.spec, diagnostic_approval=plan) as permit:
            permit.invoke(Provider(failure), 'generator', 0, request())

    def test_one_way_history_and_unused_allowance_preserved(self):
        preview, receipt = self.ready()
        current = json.loads(self.destination.path.read_bytes())
        original = json.loads(self.original)
        self.assertEqual(current['workflows'], original['workflows'])
        self.assertEqual(current['pricing_revisions'], original['pricing_revisions'])
        self.assertEqual(current['legacy'], original['legacy'])
        self.assertEqual((current['max_attempts'], current['max_reserved_usd']), (12, '0.100000'))
        self.assertEqual((preview.artifact_directory/'source.v2.json').read_bytes(), self.original)
        self.assertEqual((self.source.directory/WINDOW_NAME).read_bytes(), self.window)
        self.assertEqual(self.source.lock_path.stat().st_ino, json.loads(
            (preview.artifact_directory/'plan.json').read_bytes())['source_identity']['lock']['inode'])
        with self.assertRaises(BudgetError):
            with self.source.reserve(self.spec, diagnostic_approval=self.previous): self.fail('old authority')
        # Even a reviewed writer unaware of the new marker rejects the tombstone schema.
        with self.assertRaises(BudgetError):
            self.source._validate_document(json.loads(self.source.path.read_bytes()))
        self.invoke(preview.plan_sha256)
        after = json.loads(self.destination.path.read_bytes())
        self.assertEqual(after['workflows'][:7], original['workflows'])
        self.assertEqual(len(after['workflows']), 8)
        self.assertEqual(after['workflows'][-1]['status'], 'closed')

    def test_original_unrenewed_three_link_chain_succeeds(self):
        # Select the earlier synthetic checkpoint, not a production rollback.
        self.f.now -= timedelta(days=2)
        self.original, self.window, self.previous = self.f.source, self.f.window, self.f.previous
        for budget in (self.source, self.destination):
            budget.path.write_bytes(self.original)
            (budget.directory/WINDOW_NAME).write_bytes(self.window)
        preview, _ = self.ready()
        chain = json.loads((preview.artifact_directory/'chain.json').read_bytes())
        self.assertEqual(len(chain), 3)
        self.invoke(preview.plan_sha256)
        value = json.loads(self.destination.path.read_bytes())
        original = json.loads(self.original)
        self.assertEqual(value['pricing_revisions'], original['pricing_revisions'])
        self.assertEqual(value['workflows'][:7], original['workflows'])
        self.assertEqual(len(value['workflows']), 8)

    def test_duplicate_restart_and_changed_id_are_denied(self):
        preview, _ = self.ready(); self.invoke(preview.plan_sha256)
        self.destination = PersistentBudget(self.source.root, scope=self.source.scope,
            billing_scope=self.source.billing_scope, policy=BudgetPolicy(12), clock=lambda: self.f.now,
            authority_root=self.destination_root, authorized_authority_root=self.destination_root,
            before_request=live_access.require_local_access)
        for spec in (self.spec, replace(self.spec, invocation_id='another')):
            with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256, spec)
        self.assertEqual(len(json.loads(self.destination.path.read_bytes())['workflows']), 8)

    def test_prepare_has_no_authority_write_and_destination_cannot_activate_early(self):
        preview = self.prepare(dry_run=True)
        self.assertIsNone(preview.artifact_directory)
        self.assertEqual(self.source.path.read_bytes(), self.original)
        self.assertEqual(self.destination.path.read_bytes(), self.original)
        preview = self.prepare(); self.transfer(preview)
        with self.assertRaises((BudgetError, OSError)):
            self.activate(preview, '0'*64)
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)
        # Original hosted batch remains usable until retirement actually begins.
        from northstar_atlas.demo_batch import remaining
        self.assertEqual(remaining(self.source, self.previous), 5)

    def test_expiry_mismatch_and_missing_approval_deny_before_provider(self):
        preview, _ = self.ready()
        for spec in (replace(self.spec, question='Different question'), replace(self.spec, backend='chroma'),
                     replace(self.spec, mode='draft_review'), replace(self.spec, profile='llm_routing_grading_semantic_v1'),
                     replace(self.spec, reference_date=self.spec.reference_date+timedelta(days=1))):
            with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256, spec)
        with self.assertRaises(BudgetError): self.invoke(self.previous)
        with self.assertRaises(BudgetError):
            with self.destination.reserve(self.spec): self.fail('no approval')
        self.f.now += timedelta(minutes=31)
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)
        self.assertEqual(len(json.loads(self.destination.path.read_bytes())['workflows']), 7)

    def test_failed_and_zero_call_attempts_cannot_reopen_the_permit(self):
        preview, _ = self.ready()
        with self.assertRaises(TimeoutError): self.invoke(preview.plan_sha256, failure=TimeoutError())
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)
        current = json.loads(self.destination.path.read_bytes())
        self.assertEqual(current['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(current['workflows'][:7], json.loads(self.original)['workflows'])

    def test_local_only_successor_rejects_missing_browser_and_hosted_context_before_reservation(self):
        preview, _ = self.ready()
        for kind in ('missing-browser', 'hosted', 'no-access-verifier'):
            with self.subTest(kind=kind):
                old = self.client
                if kind == 'missing-browser': self.client = None
                if kind == 'hosted': os.environ['REPLIT_DEV_DOMAIN'] = 'synthetic.replit.dev'
                if kind == 'no-access-verifier': self.destination.before_request = None
                try:
                    with self.assertRaises((BudgetError, live_access.LocalAccessError)):
                        with self.destination.reserve(self.spec, diagnostic_approval=preview.plan_sha256):
                            self.fail('unauthorized reservation')
                finally:
                    self.client = old; os.environ.pop('REPLIT_DEV_DOMAIN', None)
                    self.destination.before_request = live_access.require_local_access
        self.assertEqual(len(json.loads(self.destination.path.read_bytes())['workflows']), 7)

    def test_activation_alone_never_authorizes_paid_execution(self):
        preview, _ = self.ready()
        os.environ.pop('NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL')
        with self.assertRaises(BudgetError):
            with self.destination.reserve(self.spec, diagnostic_approval=preview.plan_sha256):
                self.fail('activation was mistaken for execution approval')
        os.environ['NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL'] = '0'*64
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)
        self.assertEqual(len(json.loads(self.destination.path.read_bytes())['workflows']), 7)

    def test_expired_price_before_prepare_and_expired_plan_before_retire(self):
        self.f.now += timedelta(hours=2)
        with self.assertRaises(BudgetError): self.prepare()
        self.f.now -= timedelta(hours=2)
        preview = self.prepare(); self.f.now += timedelta(minutes=31)
        with self.assertRaises(BudgetError): self.retire(preview)
        self.assertEqual(self.source.path.read_bytes(), self.original)

    def test_expired_destination_activation_leaves_source_retired(self):
        preview = self.prepare(); receipt = self.retire(preview); self.transfer(preview)
        self.f.now += timedelta(minutes=31)
        with self.assertRaises(BudgetError): self.activate(preview, receipt)
        with self.assertRaises(BudgetError): self.retire(preview)
        self.assertEqual(self.destination.path.read_bytes(), self.original)

    def test_writers_and_destination_evidence_are_not_assumed(self):
        for change in (dict(writers_stopped=False), dict(approved_destination_sha256='0'*64),
                       dict(reviewed_writer_evidence_sha256=''), dict(dry_run='yes')):
            with self.subTest(change=change), self.assertRaises(BudgetError): self.prepare(**change)
        self.assertEqual(self.source.path.read_bytes(), self.original)

    def test_device_only_change_is_not_accepted_as_inode_match(self):
        m = self.module(); preview = self.prepare(); identity = m._identity
        def changed(budget):
            value = identity(budget); value['lock']['device'] += 1
            return value
        with patch.object(m, '_identity', side_effect=changed), self.assertRaises(BudgetError): self.retire(preview)
        self.assertEqual(self.source.path.read_bytes(), self.original)

    def test_unfinished_or_partially_consumed_source_cannot_handoff(self):
        spec = replace(self.spec, invocation_id='source-submission')
        with self.source.reserve(spec, diagnostic_approval=self.previous): pass
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.destination.path.read_bytes(), self.original)

    def test_hard_process_exit_preserves_uncertain_full_reservation(self):
        preview, _ = self.ready()
        ctx = multiprocessing.get_context('fork')
        def interrupted():
            with self.destination.reserve(self.spec, diagnostic_approval=preview.plan_sha256): os._exit(17)
        process = ctx.Process(target=interrupted); process.start(); process.join(15)
        self.assertEqual(process.exitcode, 17)
        value = json.loads(self.destination.path.read_bytes())
        self.assertEqual(value['workflows'][-1]['status'], 'reserved')
        self.assertEqual(self.destination.inspect_history(diagnostic_approval=preview.plan_sha256).attempts_consumed, 8)
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)

    def test_retirement_waits_for_existing_exclusive_lock(self):
        preview = self.prepare(); ctx=multiprocessing.get_context('fork'); parent, child=ctx.Pipe()
        def attempt():
            try: self.retire(preview)
            except BudgetError: child.send('denied')
            else: child.send('unsafe')
        with self.source._physical_lock():
            process=ctx.Process(target=attempt); process.start()
            self.assertTrue(parent.poll(15)); self.assertEqual(parent.recv(), 'denied')
            process.join(15); self.assertEqual(process.exitcode, 0)
        parent.close(); child.close()
        self.assertEqual(self.source.path.read_bytes(), self.original)

    def test_hard_exit_after_source_replacement_never_reactivates_either_side(self):
        preview=self.prepare(); ctx=multiprocessing.get_context('fork')
        def interrupted():
            original = self.source._write
            def crash(value): original(value); os._exit(23)
            self.source._write = crash
            self.retire(preview)
        process=ctx.Process(target=interrupted); process.start(); process.join(15)
        self.assertEqual(process.exitcode, 23)
        self.assertTrue(self.source.pending_path.exists())
        self.assertFalse((preview.artifact_directory/'retirement.json').exists())
        self.transfer(preview)
        with self.assertRaises(BudgetError): self.activate(preview,'0'*64)
        with self.assertRaises(BudgetError):
            with self.source.reserve(self.spec,diagnostic_approval=self.previous): self.fail('source reopened')

    def test_activation_receipt_failure_keeps_guard(self):
        m = self.module(); preview = self.prepare(); receipt = self.retire(preview); self.transfer(preview)
        write = m._write_private
        def fail(budget, path, raw):
            if path.name == 'activation.completed': raise OSError('synthetic persistence failure')
            return write(budget, path, raw)
        with patch.object(m, '_write_private', side_effect=fail), self.assertRaises(BudgetError):
            self.activate(preview, receipt)
        self.assertTrue(self.destination.pending_path.exists())
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)

    def test_wrong_source_destination_and_approval_denied(self):
        preview = self.prepare()
        with self.assertRaises(BudgetError): self.retire(preview, approved_plan_sha256='0'*64)
        with self.assertRaises(BudgetError): self.retire(preview, writers_stopped=False)
        with self.assertRaises(BudgetError): self.retire(preview, reviewed_writer_evidence_sha256='0'*64)
        original = self.source.path
        other = original.parent/'replacement.json'; other.write_bytes(self.original); other.chmod(0o600)
        os.replace(other, original)
        with self.assertRaises(BudgetError): self.retire(preview)
        self.assertFalse((self.source.directory/self.module().RETIRED).exists())

    def test_destination_identity_changed_after_prepare_denied(self):
        preview = self.prepare(); receipt = self.retire(preview); self.transfer(preview)
        path = self.destination.lock_path
        path.rename(path.with_name('old.lock')); path.touch(mode=0o600)
        with self.assertRaises(BudgetError): self.activate(preview, receipt)
        self.assertEqual(self.destination.path.read_bytes(), self.original)

    def test_original_bytes_and_approval_chain_changed_denied(self):
        preview = self.prepare()
        self.source.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError): self.retire(preview)
        self.source.path.write_bytes(self.original)
        chain = preview.artifact_directory/'chain.json'; original = chain.read_bytes()
        chain.write_bytes(original+b' ')
        with self.assertRaises(BudgetError): self.retire(preview)
        chain.write_bytes(original)
        (self.source.directory/WINDOW_NAME).write_bytes(self.window+b' ')
        with self.assertRaises(BudgetError): self.retire(preview)

    def test_destination_tamper_receipt_tamper_and_reapplication_denied(self):
        preview = self.prepare(); receipt = self.retire(preview); target = self.transfer(preview)
        with self.assertRaises(BudgetError): self.activate(preview, receipt, approved_retirement_sha256='0'*64)
        path = target/'candidate.v2.json'; original = path.read_bytes(); path.write_bytes(original+b' ')
        with self.assertRaises(BudgetError): self.activate(preview, receipt)
        path.write_bytes(original)
        self.destination.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError): self.activate(preview, receipt)
        self.destination.path.write_bytes(self.original)
        self.activate(preview, receipt)
        with self.assertRaises(BudgetError): self.activate(preview, receipt)
        with self.assertRaises(BudgetError): self.retire(preview)

    def test_pending_settlement_and_unfinished_history_denied(self):
        self.source._mark_settlement()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.destination.path.read_bytes(), self.original)

    def test_source_write_failure_never_produces_retirement_receipt(self):
        preview = self.prepare()
        with patch.object(self.source, '_write', side_effect=OSError('synthetic')), self.assertRaises(BudgetError):
            self.retire(preview)
        self.assertTrue(self.source.pending_path.exists())
        self.assertTrue((self.source.directory/self.module().RETIRED).exists())
        self.assertFalse((preview.artifact_directory/'retirement.json').exists())
        self.transfer(preview)
        with self.assertRaises(BudgetError): self.activate(preview, '0'*64)
        with self.assertRaises(BudgetError):
            with self.source.reserve(self.spec, diagnostic_approval=self.previous): self.fail('reactivation')

    def test_destination_write_failure_cannot_restore_source_or_retry(self):
        preview = self.prepare(); receipt = self.retire(preview); self.transfer(preview)
        with patch.object(self.destination, '_write', side_effect=OSError('synthetic')), self.assertRaises(BudgetError):
            self.activate(preview, receipt)
        self.assertTrue(self.destination.pending_path.exists())
        with self.assertRaises(BudgetError): self.activate(preview, receipt)
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)
        # Restoring only source bytes cannot reopen an authority with its durable fence.
        self.source.path.write_bytes(self.original)
        fresh = PersistentBudget(self.source.root, scope=self.source.scope, billing_scope=self.source.billing_scope,
                                 policy=BudgetPolicy(12), clock=lambda:self.f.now)
        with self.assertRaises(BudgetError):
            with fresh.reserve(self.spec, diagnostic_approval=self.previous): self.fail('restored source')

    def test_zero_transport_interruption_still_burns_single_invocation(self):
        preview, _ = self.ready()
        with self.assertRaises(KeyboardInterrupt):
            with self.destination.reserve(self.spec, diagnostic_approval=preview.plan_sha256):
                raise KeyboardInterrupt()
        with self.assertRaises(BudgetError): self.invoke(preview.plan_sha256)
        self.assertEqual(len(json.loads(self.destination.path.read_bytes())['workflows']), 8)

    def test_two_processes_and_existing_lock_exclude_competing_writers(self):
        preview, _ = self.ready()
        ctx = multiprocessing.get_context('fork'); parent, child = ctx.Pipe()
        def attempt():
            try: self.invoke(preview.plan_sha256)
            except BudgetError: child.send('denied')
            else: child.send('reserved')
        processes = [ctx.Process(target=attempt) for _ in range(2)]
        for p in processes: p.start()
        results=[]
        for _ in processes:
            self.assertTrue(parent.poll(15)); results.append(parent.recv())
        for p in processes:
            p.join(15); self.assertEqual(p.exitcode, 0)
        parent.close(); child.close()
        self.assertEqual(sorted(results), ['denied','reserved'])
        self.assertEqual(len(json.loads(self.destination.path.read_bytes())['workflows']), 8)

    def test_private_permissions_symlinks_and_missing_destination_denied(self):
        self.destination.path.chmod(0o644)
        with self.assertRaises(BudgetError): self.prepare()
        self.destination.path.chmod(0o600)
        path = self.destination.path; path.rename(path.with_name('original.json'))
        path.symlink_to(path.with_name('original.json'))
        with self.assertRaises(BudgetError): self.prepare()
        path.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse(path.exists())

    def test_run_approval_consent_settlement_and_recording_order(self):
        from test_result_capture import success
        from northstar_atlas.llm import GenerationLimits
        m = self.module(); preview, _ = self.ready(); calls=[]
        @contextmanager
        def opener(root, **options):
            self.assertEqual(json.loads(self.destination.path.read_bytes())['workflows'][-1]['status'], 'reserved')
            def ask(question, **kw):
                calls.append(question)
                options['reservation'].invoke(Provider(), 'generator', 0, request())
                return success()
            yield SimpleNamespace(ask=ask)
        options = dict(plan_sha256=preview.plan_sha256, approved_run_sha256=m.run_approval(preview.plan_sha256),
                       consent=True, opener=opener, limits=GenerationLimits())
        with self.assertRaises(BudgetError):
            m.run_handoff_question(self.destination, self.spec, **dict(options, consent=False))
        with self.assertRaises(BudgetError):
            m.run_handoff_question(self.destination, self.spec, **dict(options, approved_run_sha256='0'*64))
        result = m.run_handoff_question(self.destination, self.spec, **options)
        self.assertEqual(result.public.kind, 'answer')
        self.assertEqual(json.loads(self.destination.path.read_bytes())['workflows'][-1]['status'], 'closed')
        record = self.source.root/f'.local/linear-demo-results/{self.spec.invocation_id}.json'
        self.assertEqual(json.loads(record.read_bytes())['status'], 'accepted')
        with self.assertRaises((FileExistsError, BudgetError)):
            m.run_handoff_question(self.destination, self.spec, **options)
        self.assertEqual(calls, [self.spec.question])


if __name__ == '__main__':
    unittest.main()
