"""No-upsert restoration: synthetic publications, actual locks, fake HTTP only."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import test_pinecone_diagnostic_audit as fixtures
from northstar_atlas.vector_store import pinecone_workflow as workflow
from northstar_atlas.vector_store.records import VectorStoreError


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.vector_store.pinecone_restoration'),
                             'guarded restoration implementation is missing')
        from northstar_atlas.vector_store import pinecone_restoration as restoration
        self.r = restoration
        fixtures.DiagnosticTests.setUp(self)
        shutil.copytree(fixtures.ROOT / 'data', self.root / 'data')
        target = {k: getattr(self.settings, k) for k in
                  ('name', 'cloud', 'region', 'organization', 'project_name', 'project_id', 'plan')}
        (self.root / '.local/pinecone/target.json').write_text(json.dumps(target))
        self.publication_id = self.store.state.path.name
        self.now = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
        self.before = [p.read_bytes() for p in self.paths]
        self.pointer_identity = self.paths[1].stat().st_ino
        self.visible = []

    def send(self, request):
        if request.url.path.endswith('/search'):
            self.calls.append('search')
            if self.failure:
                value = self.failure('search', request)
                if value is not None: return value
            body = json.loads(request.content)
            key = body['filter']['record_id']['$eq']
            self.assertEqual(body['filter'], {'record_id': {'$eq': key}})
            self.assertEqual(body['top_k'], 1)
            self.assertEqual(body['score_by'], [{'type': 'dense_vector', 'field': 'embedding',
                                               'values': list(self.store.by_id[key].vector)}])
            self.visible.append(key)
            row = {k: v for k, v in self.store.wire[key].items() if k != 'embedding'}
            return httpx.Response(200, json={'namespace': self.snapshot.namespace,
                'matches': [{**row, '_score': 1.0}], 'usage': {'read_units': 2, 'egress_bytes': 10}})
        return fixtures.DiagnosticTests.send(self, request)

    def prepare(self):
        self.plan = self.r.prepare_restoration(self.root, self.publication_id,
                                              approve_preparation=True, now=self.now)
        self.assertEqual(self.before, [p.read_bytes() for p in self.paths])
        self.assertEqual(self.calls, [])
        return self.plan

    def apply(self, **kwargs):
        return self.r.apply_restoration(self.root, self.publication_id,
            approved_plan_sha256=self.plan['plan_sha256'], approve_promotion=True,
            settings=self.settings, control_transport=httpx.MockTransport(self.send),
            documents_transport=httpx.MockTransport(self.send), now=self.now, **kwargs)

    def assert_blocked(self):
        self.assertTrue((self.store.state.path / 'restoration.pending').exists())
        with self.assertRaises(VectorStoreError): self.store.status()
        with self.assertRaises(VectorStoreError): self.store.audit()
        with self.assertRaises(VectorStoreError): self.store.synchronize()
        self.assertEqual(self.paths[1].read_bytes(), self.before[1])
        self.assertEqual(self.paths[1].stat().st_ino, self.pointer_identity)

    def test_success_verifies_all_records_and_promotes_only_manifest(self):
        self.prepare(); self.egress = lambda op: 10
        report = self.apply()
        self.assertEqual(report['outcome'], 'success')
        self.assertEqual(report['attempts'], {'describe': 2, 'list': 2, 'fetch': 4, 'search': 189})
        self.assertEqual(self.calls, ['describe'] + ['list']*2 + ['fetch']*4 + ['search']*189 + ['describe'])
        self.assertEqual(set(self.visible), set(self.store.by_id))
        self.assertEqual(report['verified_records'], 189)
        self.assertEqual(report['visible_records'], 189)
        self.assertEqual(report['validated_read_units'], 384)
        self.assertEqual(report['validated_egress_bytes'], 1950)
        self.assertTrue(report['egress_complete'])
        self.assertEqual(self.store.status()['state'], 'ready')
        self.assertFalse((self.store.state.path / 'restoration.pending').exists())
        self.assertEqual(self.paths[1].read_bytes(), self.before[1])
        self.assertEqual(self.paths[1].stat().st_ino, self.pointer_identity)
        for secret in (fixtures.PRIVATE, fixtures.HOST, self.settings.api_key, self.snapshot.namespace):
            self.assertNotIn(secret, json.dumps(report))
        before_calls = len(self.calls)
        with self.assertRaises(VectorStoreError): self.apply()
        self.assertEqual(len(self.calls), before_calls)

    def test_missing_or_wrong_visibility_withholds_readiness_without_retry(self):
        self.prepare()
        self.failure = lambda op, req: httpx.Response(200, json={'namespace': self.snapshot.namespace,
            'matches': [], 'usage': {'read_units': 1}}) if op == 'search' else None
        result = self.apply()
        self.assertEqual(result['outcome'], 'failed')
        self.assertEqual(self.calls.count('search'), 1)
        self.assertEqual(self.before, [p.read_bytes() for p in self.paths])
        self.assert_blocked()

    def test_corrupted_fetch_stops_before_any_search(self):
        self.prepare()
        def corrupt(op, req):
            if op != 'fetch': return
            ids = json.loads(req.content)['ids']; rows = {i: deepcopy(self.store.wire[i]) for i in ids}
            rows[ids[0]]['text'] = fixtures.PRIVATE
            return httpx.Response(200, json={'namespace': self.snapshot.namespace,
                'documents': rows, 'usage': {'read_units': 1}})
        self.failure = corrupt
        self.assertEqual(self.apply()['outcome'], 'failed')
        self.assertNotIn('search', self.calls)
        self.assert_blocked()

    def test_wrong_approval_and_missing_promotion_consent_have_zero_io(self):
        self.prepare()
        for digest, consent in [('0'*64, True), (self.plan['plan_sha256'], False)]:
            with self.assertRaises(VectorStoreError):
                self.r.apply_restoration(self.root, self.publication_id,
                    approved_plan_sha256=digest, approve_promotion=consent, now=self.now)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.store.state.path / 'restoration.pending').exists())

    def test_expired_approval_rejected_before_credentials_or_transport(self):
        self.prepare()
        with patch.object(self.r, 'load_settings', side_effect=AssertionError('credentials forbidden')):
            with self.assertRaises(VectorStoreError):
                self.r.apply_restoration(self.root, self.publication_id,
                    approved_plan_sha256=self.plan['plan_sha256'], approve_promotion=True,
                    now=self.now + timedelta(days=1))

    def test_stale_manifest_approval_rejected_before_io(self):
        self.prepare()
        self.paths[0].write_bytes(self.before[0] + b' ')
        with self.assertRaises(VectorStoreError): self.apply()
        self.assertEqual(self.calls, [])

    def test_budget_exhaustion_blocks_201st_transport(self):
        self.prepare(); self.page_size = 1
        result = self.apply()
        self.assertEqual(result['reason'], 'budget_exhausted')
        self.assertEqual(sum(result['attempts'].values()), 200)
        self.assertEqual(len(self.calls), 200)
        self.assert_blocked()

    def test_lock_contention_prevents_credentials_and_dispatch(self):
        self.prepare()
        with self.store.state.lock(), patch.object(self.r, 'load_settings', side_effect=AssertionError('no credentials')):
            with self.assertRaises(VectorStoreError): self.apply()
        self.assertEqual(self.calls, [])

    def test_manifest_write_failure_retains_guard_and_explicit_abort_is_incomplete(self):
        self.prepare()
        with patch.object(type(self.store.state), 'write_run', side_effect=OSError('private failure')):
            self.assertEqual(self.apply()['outcome'], 'inconclusive')
        self.assertEqual(self.before, [p.read_bytes() for p in self.paths]); self.assert_blocked()
        guard = self.store.state.path / 'restoration.pending'
        digest = hashlib.sha256(guard.read_bytes()).hexdigest()
        self.r.abort_restoration(self.root, self.publication_id,
            approved_plan_sha256=self.plan['plan_sha256'], approved_guard_sha256=digest, approve_abort=True)
        self.assertFalse(guard.exists()); self.assertEqual(self.store.status()['state'], 'incomplete')
        before_calls = len(self.calls)
        with self.assertRaises(VectorStoreError): self.apply()
        self.assertEqual(len(self.calls), before_calls)

    def test_crash_after_manifest_replace_cannot_be_automatically_recovered(self):
        self.prepare(); original = type(self.store.state).write_run
        def interrupted(state, namespace, value):
            original(state, namespace, value)
            raise KeyboardInterrupt
        with patch.object(type(self.store.state), 'write_run', interrupted):
            self.assertEqual(self.apply()['outcome'], 'inconclusive')
        self.assert_blocked()
        guard = self.store.state.path / 'restoration.pending'
        with self.assertRaises(VectorStoreError):
            self.r.abort_restoration(self.root, self.publication_id,
                approved_plan_sha256=self.plan['plan_sha256'],
                approved_guard_sha256=hashlib.sha256(guard.read_bytes()).hexdigest(), approve_abort=True)
        self.assertTrue(guard.exists())

    def test_timeout_single_attempt_keeps_guard(self):
        self.prepare()
        def fail(op, req): raise httpx.ReadTimeout(fixtures.PRIVATE)
        self.failure = fail
        result = self.apply()
        self.assertEqual(result['outcome'], 'failed'); self.assertEqual(self.calls, ['describe'])
        self.assertNotIn(fixtures.PRIVATE, json.dumps(result)); self.assert_blocked()

    def test_wrong_state_cannot_prepare(self):
        for state, phase in [('ready','failed audit'),('incomplete','uploading'),('deleted','failed audit')]:
            with self.subTest(state=state):
                self.store.state.write_run(self.snapshot.namespace, self.store._manifest(state, phase=phase))
                with self.assertRaises(VectorStoreError): self.prepare()
        self.assertEqual(self.calls, [])

    def test_wrong_exact_record_identity_blocks_visibility(self):
        self.prepare()
        def wrong(op, req):
            if op != 'search': return
            requested = json.loads(req.content)['filter']['record_id']['$eq']
            key = next(k for k in self.store.wire if k != requested)
            row = {k: v for k, v in self.store.wire[key].items() if k != 'embedding'}
            return httpx.Response(200, json={'namespace': self.snapshot.namespace,
                'matches': [{**row, '_score': 1.}], 'usage': {'read_units': 1}})
        self.failure = wrong
        self.assertEqual(self.apply()['outcome'], 'failed')
        self.assertEqual(self.calls.count('search'), 1); self.assert_blocked()

    def test_final_source_change_blocks_promotion(self):
        self.prepare()
        def change(op, req):
            if op == 'search' and self.calls.count('search') == 189:
                with (self.root / 'data/policies/HR-001.md').open('a') as stream: stream.write('\nchanged')
        self.failure = change
        self.assertEqual(self.apply()['outcome'], 'failed')
        self.assertEqual(self.before, [p.read_bytes() for p in self.paths]); self.assert_blocked()

    def test_changed_pointer_bytes_block_promotion_without_repair(self):
        self.prepare()
        def change(op, req):
            if op == 'search': self.paths[1].write_bytes(self.before[1] + b' ')
        self.failure = change
        self.assertEqual(self.apply()['outcome'], 'failed')
        self.assertEqual(self.paths[0].read_bytes(), self.before[0])
        self.assertEqual(self.paths[1].read_bytes(), self.before[1] + b' ')
        with self.assertRaises(VectorStoreError): self.store.status()

    def test_failed_guard_creation_has_no_transport_and_blocks_readiness(self):
        self.prepare(); create = self.r._create
        def partial(path, raw):
            if path.name == 'restoration.pending':
                path.write_bytes(b'{')
                raise OSError('private persistence detail')
            return create(path, raw)
        with patch.object(self.r, '_create', partial), self.assertRaises(OSError): self.apply()
        self.assertEqual(self.calls, []); self.assert_blocked()

    def test_receipt_failure_leaves_ready_manifest_guarded(self):
        self.prepare(); create = self.r._create
        def fail(path, raw):
            if path.name.endswith('.receipt.json'): raise OSError('private details')
            return create(path, raw)
        with patch.object(self.r, '_create', fail):
            self.assertEqual(self.apply()['outcome'], 'inconclusive')
        self.assertEqual(json.loads(self.paths[0].read_bytes())['state'], 'ready')
        self.assert_blocked()

    def test_guard_unlink_fsync_failure_recreates_fence(self):
        self.prepare(); sync = self.r._sync; failed = []
        def fail_once(path):
            guard = self.store.state.path / 'restoration.pending'
            receipt = list(self.store.state.path.glob('*.receipt.json'))
            if path == self.store.state.path and receipt and not guard.exists() and not failed:
                failed.append(True); raise OSError('sync failed')
            return sync(path)
        with patch.object(self.r, '_sync', fail_once):
            self.assertEqual(self.apply()['outcome'], 'inconclusive')
        self.assertEqual(len(failed), 1); self.assert_blocked()

    def test_guard_cannot_be_aborted_with_unreviewed_digest(self):
        self.prepare()
        self.failure = lambda op, req: httpx.Response(503)
        self.apply()
        with self.assertRaises(VectorStoreError):
            self.r.abort_restoration(self.root, self.publication_id,
                approved_plan_sha256=self.plan['plan_sha256'], approved_guard_sha256='0'*64, approve_abort=True)
        self.assert_blocked()

    def test_tampered_immutable_plan_and_snapshot_rejected(self):
        self.prepare()
        path = self.store.state.path / ('restoration-' + self.plan['plan_sha256'] + '.json')
        original = path.read_bytes(); path.write_bytes(original + b' ')
        with self.assertRaises(VectorStoreError): self.apply()
        path.write_bytes(original)
        snap = self.root / '.local/pinecone/snapshots' / (self.snapshot.snapshot_id + '.json')
        value = json.loads(snap.read_text()); value['records'][0]['text'] += 'changed'; snap.write_text(json.dumps(value))
        with self.assertRaises(ValueError): self.apply()
        self.assertEqual(self.calls, [])

    def test_two_concurrent_transports_share_exact_budget(self):
        budget = self.r.RestorationBudget(); calls = []
        route = fixtures.HOST + '/search'
        transport = self.d.BudgetTransport(budget, {('POST', route): 'search'},
            httpx.MockTransport(lambda req: calls.append(req.method) or httpx.Response(200)))
        def send(i):
            try: transport.handle_request(httpx.Request('POST', route)); return True
            except self.d.DiagnosticError: return False
        with ThreadPoolExecutor(max_workers=8) as pool: results = list(pool.map(send, range(230)))
        self.assertEqual(sum(results), 200); self.assertEqual(len(calls), 200)

    def test_cli_preparation_requires_explicit_approval_without_credentials(self):
        from northstar_atlas import pinecone
        out = io.StringIO()
        with patch.object(pinecone, 'validate_project_root', return_value=self.root), \
                patch.object(pinecone, 'load_settings', side_effect=AssertionError('credentials forbidden')), \
                patch.object(pinecone, 'sdk_client', side_effect=AssertionError('SDK forbidden')), \
                redirect_stdout(out), redirect_stderr(out):
            result = pinecone.main(['prepare-restoration', '--publication-id', self.publication_id,
                                   '--approve-preparation'])
        self.assertEqual(result, 0)
        plan = json.loads(out.getvalue()); self.assertRegex(plan['plan_sha256'], '^[a-f0-9]{64}$')
        self.assertEqual(self.calls, []); self.assertEqual(self.before, [p.read_bytes() for p in self.paths])

    def test_status_cannot_report_ready_if_guard_appears_during_read(self):
        read = self.store.state.read_run
        def interleaved(namespace):
            (self.store.state.path / 'restoration.pending').write_text('synthetic pending transition')
            self.store.state.write_run(namespace, self.store._manifest('ready'))
            return read(namespace)
        with patch.object(self.store.state, 'read_run', interleaved):
            with self.assertRaises(VectorStoreError): self.store.status()

    def test_lock_replacement_during_verification_blocks_promotion(self):
        self.prepare()
        def replace_lock(op, request):
            if op == 'search' and self.calls.count('search') == 189:
                lock = self.store.state.path / 'writer.lock'
                lock.rename(lock.with_name('original-lock'))
                lock.write_text('')
        self.failure = replace_lock
        self.assertEqual(self.apply()['outcome'], 'failed')
        self.assertEqual(self.before, [p.read_bytes() for p in self.paths]); self.assert_blocked()

    def test_source_change_during_manifest_write_keeps_guard(self):
        self.prepare(); write = type(self.store.state).write_run
        def changed(state, namespace, value):
            write(state, namespace, value)
            with (self.root / 'data/policies/HR-001.md').open('a') as stream: stream.write('\nchanged')
        with patch.object(type(self.store.state), 'write_run', changed):
            self.assertEqual(self.apply()['outcome'], 'failed')
        self.assert_blocked()

    def test_candidate_tampering_during_report_persistence_keeps_guard(self):
        self.prepare(); save = workflow._save_private_report
        def changed(root, suffix, report):
            path = save(root, suffix, report)
            if report['outcome'] == 'success': self.paths[0].write_bytes(self.paths[0].read_bytes() + b' ')
            return path
        with patch.object(workflow, '_save_private_report', changed):
            self.assertEqual(self.apply()['outcome'], 'failed')
        self.assert_blocked()

    def test_expiry_during_report_persistence_keeps_guard(self):
        self.prepare(); save = workflow._save_private_report; clock = [self.now]
        def changed(root, suffix, report):
            path = save(root, suffix, report)
            if report['outcome'] == 'success': clock[0] += timedelta(days=1)
            return path
        with patch.object(workflow, '_save_private_report', changed), patch.object(self.r, '_now', lambda _: clock[0]):
            self.assertEqual(self.apply()['outcome'], 'failed')
        self.assert_blocked()


if __name__ == '__main__': unittest.main()
