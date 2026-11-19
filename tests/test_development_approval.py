"""Adversarial monthly approval packets; all paths are synthetic checkouts."""
import json
from datetime import datetime,timezone
from unittest.mock import patch
import unittest
import test_local_development as fixtures
from northstar_atlas.budget_transition import _encode,_sha


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.DevelopmentTests('runTest'); f.setUp(); self.addCleanup(f.doCleanups); self.f=f

    def test_invalid_pricing_never_prepares_authority(self):
        f=self.f
        for changes in ({'source':'https://unreviewed.invalid/prices'}, {'confirmed_at':'2028-01-31T12:00:00'},
                        {'valid_until':'2028-03-01T12:00:00+00:00'}, {'input_per_million_usd':'0'}):
            price=dict(f.price,**changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                f.dev.prepare(f.root,price,verified_pricing_sha256=_sha(_encode(price)),scope_review_sha256='a'*64,
                              approve_preparation=True,writers_stopped=True,clock=lambda:f.now)
            self.assertFalse((f.root/f.dev.ROOT).exists())

    def test_expired_price_cannot_activate(self):
        f=self.f; f.prepare(); b=f.dev.open_review(f.root,clock=lambda:f.now)
        f.now=datetime(2028,2,29,12,0,tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            f.dev.apply(b,approved_plan_sha256=f.preview.plan_sha256,approved_candidate_sha256=f.preview.candidate_sha256,
                        approve_application=True,writers_stopped=True)
        self.assertTrue((b.directory/f.dev.PREPARING).exists())
        self.assertFalse((b.directory/f.dev.ACTIVATION).exists())

    def test_replaced_lock_invalidates_new_process_approval(self):
        f=self.f; b=f.activate()
        b.lock_path.rename(b.lock_path.with_name('prior.lock'))
        b.lock_path.write_bytes(b''); b.lock_path.chmod(0o600)
        fresh=f.dev.open_review(f.root,approval=b.approval,run_approval=b.run_approval,clock=lambda:f.now)
        with self.assertRaises(ValueError): fresh.capacity()

    def test_pending_workflow_blocks_renewal_without_refund(self):
        f=self.f; b=f.activate(); f.authorize_local()
        # Simulate process death before settlement, retaining the initial durable reserve.
        with patch.object(b,'_settle',side_effect=OSError('synthetic interruption')),self.assertRaises(OSError):
            with b.reserve(f.spec()): pass
        raw=b.path.read_bytes(); f.now=datetime(2028,3,1,12,0,tzinfo=timezone.utc)
        price=dict(f.price,id='new',confirmed_at=f.now.isoformat(),valid_until='2028-04-01T12:00:00+00:00')
        with self.assertRaises(ValueError):
            f.dev.prepare_renewal(b,price,previous_plan_sha256=b.approval,verified_pricing_sha256=_sha(_encode(price)),
                approve_preparation=True,writers_stopped=True)
        self.assertEqual(b.path.read_bytes(),raw)
        self.assertFalse((b.directory/'review-0001').exists())

    def test_source_byte_change_invalidates_prepared_renewal(self):
        f=self.f; b=f.activate()
        f.now=datetime(2028,3,1,12,0,tzinfo=timezone.utc)
        price=dict(f.price,id='new',confirmed_at=f.now.isoformat(),valid_until='2028-04-01T12:00:00+00:00')
        preview=f.dev.prepare_renewal(b,price,previous_plan_sha256=b.approval,verified_pricing_sha256=_sha(_encode(price)),
            approve_preparation=True,writers_stopped=True)
        b.path.write_bytes(b.path.read_bytes()+b' ')
        raw=b.path.read_bytes()
        with self.assertRaises(ValueError):
            f.dev.apply(b,approved_plan_sha256=preview.plan_sha256,approved_candidate_sha256=preview.candidate_sha256,
                approve_application=True,writers_stopped=True)
        self.assertEqual(b.path.read_bytes(),raw)

    def test_repeated_price_id_and_missing_review_approval_denied(self):
        f=self.f; b=f.activate()
        for approved in (False,True):
            with self.assertRaises(ValueError):
                f.dev.prepare_renewal(b,f.price,previous_plan_sha256=b.approval,
                    verified_pricing_sha256=_sha(_encode(f.price)),approve_preparation=approved,writers_stopped=True)
        self.assertFalse((b.directory/'review-0001').exists())

    def test_bad_run_approval_never_opens_resources(self):
        f=self.f; b=f.activate(); f.authorize_local()
        b.run_approval='a'*64
        with self.assertRaises(ValueError):
            with b.reserve(f.spec()): self.fail('unauthorized reservation')
        self.assertEqual(json.loads(b.path.read_bytes())['workflows'],[])

    def test_receipt_write_failure_blocks_all_new_execution(self):
        f=self.f; f.prepare(); b=f.dev.open_review(f.root,clock=lambda:f.now)
        writer=f.dev._write_private
        def fail_receipt(budget,path,raw):
            if path.name=='applied.json': raise OSError('synthetic receipt failure')
            return writer(budget,path,raw)
        with patch.object(f.dev,'_write_private',side_effect=fail_receipt),self.assertRaises(OSError):
            f.dev.apply(b,approved_plan_sha256=f.preview.plan_sha256,approved_candidate_sha256=f.preview.candidate_sha256,
                        approve_application=True,writers_stopped=True)
        self.assertTrue(b.pending_path.exists())
        fresh=f.dev.open_review(f.root,approval=f.preview.plan_sha256,
                               run_approval=f.dev.run_approval(f.preview.plan_sha256),clock=lambda:f.now)
        with self.assertRaises(ValueError): fresh.capacity()
