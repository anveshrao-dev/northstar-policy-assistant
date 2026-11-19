"""Bounded diagnostic audits use synthetic publications and blocked HTTP transports."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor
import io
import importlib.util
import json
import tempfile
import unittest

import httpx

from test_vector_store import ROOT
from test_pinecone_cloud import cloud_settings
from northstar_atlas.embeddings import EmbeddedChunk, EmbeddingModel
from northstar_atlas.ingestion.document_loader import load_documents
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.local_embeddings import MODEL_ID, MODEL_REVISION
from northstar_atlas.vector_store.records import IndexConfig, VectorStoreError
from northstar_atlas.vector_store.preparation import freeze_snapshot
from northstar_atlas.vector_store.pinecone_store import PineconeStore, PineconeTarget
from northstar_atlas.vector_store.pinecone_cloud import record_confirmation
from northstar_atlas.vector_store import pinecone_workflow as workflow

HOST = 'https://synthetic.svc.pinecone.io'
PRIVATE = 'synthetic-private-sentinel'


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.vector_store.pinecone_diagnostic'),
                             'diagnostic-only audit implementation is missing')
        from northstar_atlas.vector_store import pinecone_diagnostic as diagnostic
        self.d = diagnostic
        temp = tempfile.TemporaryDirectory(dir=ROOT / '.local/tmp', prefix='diagnostic-audit-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'pyproject.toml').write_text('[project]\nname="northstar-atlas"\n')
        self.settings = cloud_settings()
        record_confirmation(self.root, self.settings, project_id=self.settings.project_id)
        model = EmbeddingModel(MODEL_ID, MODEL_REVISION, 384, 'l2')
        records = []
        for doc in load_documents(ROOT, mode='draft_review'):
            chunk = chunk_document(doc).chunks[0]
            for i in range(workflow.EXPECTED_COUNTS[doc.document_id]):
                records.append(EmbeddedChunk(replace(chunk, chunk_id=f'{doc.document_id}-synthetic-{i}'),
                                             (1.,) + (0.,) * 383, model))
        self.snapshot = freeze_snapshot(records, IndexConfig(model, 256), workflow.EXPECTED_CORPUS)
        workflow.save_snapshot(self.root, self.snapshot)
        self.target = PineconeTarget(self.settings.name, HOST, self.settings.project_id)
        self.description = {'name': self.settings.name, 'host': HOST.removeprefix('https://'),
            'status': {'ready': True, 'state': 'Ready'}, 'deletion_protection': 'enabled',
            'deployment': {'deployment_type': 'managed', 'cloud': 'azure', 'region': 'eastus2'},
            'schema': {'fields': {'embedding': {'type': 'dense_vector', 'dimension': 384, 'metric': 'cosine'}}},
            'tags': {'optional': PRIVATE}, 'read_capacity': {'mode': 'OnDemand'}}
        self.store = PineconeStore(self.root, self.snapshot, self.target, documents=None,
            describe_target=lambda: self.description, current_fingerprint=lambda: workflow.EXPECTED_CORPUS)
        with self.store.state.lock():
            self.store.state.write_run(self.snapshot.namespace, self.store._manifest('ready'))
            self.store.state.publish(self.snapshot.namespace)
            self.store.state.write_run(self.snapshot.namespace, self.store._manifest('incomplete', phase='failed audit'))
        self.paths = [self.store.state.path / (self.snapshot.namespace + '.json'), self.store.state.path / 'active.json']
        self.calls = []
        self.failure = None
        self.page_size = 100
        self.fingerprint = workflow.EXPECTED_CORPUS
        self.egress = None

    def send(self, request):
        op = 'describe' if request.method == 'GET' else request.url.path.rsplit('/', 1)[-1]
        self.calls.append(op)
        self.assertEqual(request.headers['X-Pinecone-Api-Version'], '2026-07')
        self.assertEqual(request.headers['Api-Key'], self.settings.api_key)
        if self.failure:
            result = self.failure(op, request)
            if result is not None: return result
        if op == 'describe':
            self.assertEqual(str(request.url), 'https://api.pinecone.io/indexes/' + self.settings.name)
            return httpx.Response(200, json=self.description)
        body = json.loads(request.content)
        ids = sorted(self.store.wire)
        if op == 'list':
            start = int(body.get('pagination_token', '0'))
            value = {'documents': [{'_id': i} for i in ids[start:start+self.page_size]]}
            if start + self.page_size < len(ids): value['pagination'] = {'next': str(start+self.page_size)}
        elif op == 'fetch':
            value = {'documents': {i: self.store.wire[i] for i in body['ids']}}
        else: self.fail('forbidden remote operation')
        usage = {'read_units': 1}
        if self.egress is not None:
            egress = self.egress(op)
            if egress is not None: usage['egress_bytes'] = egress
        return httpx.Response(200, json={**value, 'namespace': self.snapshot.namespace, 'usage': usage})

    def run_audit(self):
        before = [p.read_bytes() for p in self.paths]
        with patch.object(workflow, 'corpus_snapshot', side_effect=lambda root: ([], self.fingerprint)):
            report = self.d.run_diagnostic(self.root, self.settings,
                control_transport=httpx.MockTransport(self.send), documents_transport=httpx.MockTransport(self.send))
        self.assertEqual(before, [p.read_bytes() for p in self.paths])
        raw = json.dumps(report)
        for value in (PRIVATE, self.settings.api_key, HOST, self.snapshot.namespace, self.settings.project_id):
            self.assertNotIn(value, raw)
        reports = list((self.root / '.local/pinecone/m33').glob('*-diagnostic-audit.json'))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text()), report)
        self.assertEqual(reports[0].stat().st_mode & 0o777, 0o600)
        return report

    def test_complete_189_record_audit_is_eight_dispatches_without_promotion(self):
        report = self.run_audit()
        self.assertEqual(report['outcome'], 'success')
        self.assertEqual(self.calls, ['describe']*2 + ['list']*2 + ['fetch']*4)
        self.assertEqual(report['attempts'], {'describe': 2, 'list': 2, 'fetch': 4})
        self.assertEqual(report['validated_read_units'], 6)
        self.assertTrue(report['publication_preserved'])
        self.assertEqual(self.store.status()['state'], 'incomplete')
        with self.assertRaises(VectorStoreError): self.store.audit()
        self.assertIsNone(report['validated_egress_bytes'])
        self.assertFalse(report['egress_complete'])

    def test_189_record_audit_retains_complete_egress_without_promotion(self):
        self.egress = lambda op: 10 if op == 'list' else 100
        report = self.run_audit()
        self.assertEqual(report['outcome'], 'success')
        self.assertEqual(report['validated_read_units'], 6)
        self.assertEqual(report['validated_egress_bytes'], 420)
        self.assertEqual(report['responses_with_validated_egress'], 6)
        self.assertTrue(report['egress_complete'])
        self.assertEqual(self.store.status()['state'], 'incomplete')

    def test_189_record_audit_retains_partial_egress(self):
        self.egress = lambda op: 10 if op == 'list' else None
        report = self.run_audit()
        self.assertEqual(report['outcome'], 'success')
        self.assertEqual(report['validated_egress_bytes'], 20)
        self.assertEqual(report['responses_with_validated_egress'], 2)
        self.assertFalse(report['egress_complete'])

    def test_invalid_egress_stops_audit_and_preserves_prior_usage(self):
        self.egress = lambda op: 10 if op == 'list' else PRIVATE
        report = self.run_audit()
        self.assertEqual(report['reason'], 'usage_egress_bytes')
        self.assertEqual(report['validated_read_units'], 2)
        self.assertEqual(report['validated_egress_bytes'], 20)
        self.assertFalse(report['egress_complete'])
        self.assertEqual(self.calls, ['describe']*2 + ['list']*2 + ['fetch'])

    def test_budget_exhaustion_stops_before_thirteenth_dispatch(self):
        self.page_size = 1
        report = self.run_audit()
        self.assertEqual(report['outcome'], 'inconclusive')
        self.assertEqual(report['reason'], 'budget_exhausted')
        self.assertEqual(self.calls, ['describe']*2 + ['list']*10)
        self.assertEqual(sum(report['attempts'].values()), 12)

    def test_wrong_states_do_not_dispatch_documents(self):
        for state, phase in [('ready', 'failed audit'), ('incomplete', 'uploading'), ('deleted', 'failed audit')]:
            with self.subTest(state=state, phase=phase):
                self.store.state.write_run(self.snapshot.namespace, self.store._manifest(state, phase=phase))
                # Each invocation writes an independent outcome; retain earlier evidence.
                before = [p.read_bytes() for p in self.paths]
                with patch.object(workflow, 'corpus_snapshot', return_value=([], workflow.EXPECTED_CORPUS)):
                    result = self.d.run_diagnostic(self.root, self.settings,
                        control_transport=httpx.MockTransport(self.send), documents_transport=httpx.MockTransport(self.send))
                self.assertEqual(result['outcome'], 'failed')
                self.assertEqual(before, [p.read_bytes() for p in self.paths])
        self.assertTrue(all(op == 'describe' for op in self.calls))

    def test_list_failure_retains_sanitized_diagnostic(self):
        self.failure = lambda op, req: httpx.Response(200, json={'documents': [], 'namespace': self.snapshot.namespace,
            'usage': {'read_units': 1, PRIVATE: PRIVATE}}) if op == 'list' else None
        report = self.run_audit()
        self.assertEqual(report['outcome'], 'failed')
        self.assertEqual(report['reason'], 'usage_extra_fields')
        self.assertEqual(report['http_status'], 200)
        self.assertEqual(report['page_ordinal'], 1)
        self.assertEqual(len(self.calls), 3)

    def test_control_timeout_is_one_attempt_and_no_documents(self):
        def fail(op, req): raise httpx.ReadTimeout(PRIVATE)
        self.failure = fail
        report = self.run_audit()
        self.assertEqual(report['reason'], 'transport_timeout')
        self.assertEqual(self.calls, ['describe'])
        self.assertEqual(report['operation'], 'describe')

    def test_cli_is_diagnostic_only_and_sanitized(self):
        from northstar_atlas import pinecone
        original = self.d.run_diagnostic
        def run(root):
            with patch.object(workflow, 'corpus_snapshot', return_value=([], workflow.EXPECTED_CORPUS)):
                return original(root, self.settings, control_transport=httpx.MockTransport(self.send),
                                documents_transport=httpx.MockTransport(self.send))
        out = io.StringIO()
        with patch.object(pinecone, 'validate_project_root', return_value=self.root), \
                patch.object(self.d, 'run_diagnostic', side_effect=run), \
                patch.object(pinecone, 'sdk_client', side_effect=AssertionError('SDK forbidden')), \
                patch.object(pinecone, 'MiniLMProvider', side_effect=AssertionError('model forbidden')), \
                redirect_stdout(out), redirect_stderr(out):
            result = pinecone.main(['diagnostic-audit'])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(out.getvalue())['outcome'], 'success')
        self.assertNotIn(PRIVATE, out.getvalue())

    def test_describe_host_and_schema_rejections(self):
        bad = [ {'host': host} for host in ('http://synthetic.svc.pinecone.io', 'https://https://synthetic.svc.pinecone.io',
            'https://user@synthetic.svc.pinecone.io', HOST+':443', HOST+'/path', HOST+'?x=y', HOST+'#f',
            'synthetic.pinecone.io.evil.invalid', '127.0.0.1', None)]
        bad += [{'name': 'wrong-index'}, {'status': {'ready': 1, 'state': 'Ready'}}, {'deletion_protection': 'disabled'},
                {'deployment': {'deployment_type': 'managed', 'cloud': 'aws', 'region': 'eastus2'}},
                {'schema': {'fields': {'embedding': {'type': 'dense_vector', 'dimension': 384, 'metric': 'cosine', 'embed': {'model': PRIVATE}}}}}]
        for change in bad:
            calls=[]
            def send(req): calls.append(req); return httpx.Response(200, json={**self.description, **change})
            with self.subTest(change=change), self.d.DescribeClient(self.settings, self.d.DispatchBudget(),
                    transport=httpx.MockTransport(send)) as client:
                with self.assertRaises(self.d.DiagnosticError): client.describe()
                self.assertEqual(len(calls), 1)

    def test_describe_missing_fields_and_invalid_json(self):
        bodies = [b'[]', b'null', b'not json', b'{"x":1,"x":2}', b'{"x":NaN}']
        for name in ('name','host','status','schema','deployment','deletion_protection'):
            v=deepcopy(self.description);del v[name];bodies.append(json.dumps(v).encode())
        for body in bodies:
            with self.subTest(body=body), self.d.DescribeClient(self.settings,self.d.DispatchBudget(),
                    transport=httpx.MockTransport(lambda req: httpx.Response(200, content=body))) as client:
                with self.assertRaises(self.d.DiagnosticError): client.describe()

    def test_normalized_host_retains_raw_safety_fields(self):
        for host in (HOST, HOST.removeprefix('https://')):
            with self.d.DescribeClient(self.settings,self.d.DispatchBudget(), transport=httpx.MockTransport(
                    lambda req: httpx.Response(200,json={**self.description,'host':host}))) as client:
                value=client.describe()
                self.assertEqual(value['host'],HOST)
                self.assertEqual(value['schema'],self.description['schema'])
                self.assertEqual(value['tags'],self.description['tags'])

    def test_transport_error_status_and_redirect_never_retry(self):
        cases=[301,302,307,308,400,401,403,404,429,500,502,503,504,
               httpx.ConnectTimeout,httpx.ReadTimeout,httpx.WriteTimeout,httpx.PoolTimeout,httpx.ConnectError]
        for case in cases:
            calls=[];budget=self.d.DispatchBudget()
            def send(req):
                calls.append(req)
                if isinstance(case,int): return httpx.Response(case,text=PRIVATE,headers={'location':'https://other.invalid'})
                raise case(PRIVATE)
            with self.subTest(case=case), self.d.DescribeClient(self.settings,budget,transport=httpx.MockTransport(send)) as client:
                with self.assertRaises(self.d.DiagnosticError) as caught: client.describe()
                self.assertNotIn(PRIVATE,str(caught.exception))
                self.assertEqual(len(calls),1)
                self.assertEqual(budget.counts(),{'describe':1,'list':0,'fetch':0})

    def test_atomic_budget_across_competing_transports(self):
        budget=self.d.DispatchBudget();calls=[]
        allowed={('GET','https://api.pinecone.io/indexes/test'):'describe',('POST',HOST+'/list'):'list'}
        transport=self.d.BudgetTransport(budget,allowed,httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(200)))
        def send(i):
            request=httpx.Request('GET' if i%2 else 'POST','https://api.pinecone.io/indexes/test' if i%2 else HOST+'/list')
            try: transport.handle_request(request);return True
            except self.d.DiagnosticError: return False
        with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(send,range(40)))
        self.assertEqual(sum(results),12);self.assertEqual(len(calls),12)
        self.assertEqual(sum(budget.counts().values()),12)

    def test_unexpected_transport_route_is_blocked_without_count(self):
        budget=self.d.DispatchBudget();calls=[]
        transport=self.d.BudgetTransport(budget,{},httpx.MockTransport(lambda req:calls.append(req)))
        with self.assertRaises(self.d.DiagnosticError):transport.handle_request(httpx.Request('POST',HOST+'/search'))
        self.assertEqual(calls,[]);self.assertEqual(sum(budget.counts().values()),0)

    def test_lock_contention_does_not_read_documents(self):
        with self.store.state.lock(): report=self.run_audit()
        self.assertEqual(report['outcome'],'failed')
        self.assertEqual(self.calls,['describe'])

    def test_invalid_ownership_blocks_before_second_describe(self):
        value=self.store.state.read_run(self.snapshot.namespace);value['payload_digest']='altered'
        self.store.state.write_run(self.snapshot.namespace,value)
        report=self.run_audit()
        self.assertEqual(report['reason'],'ownership');self.assertEqual(self.calls,['describe'])

    def test_target_change_between_descriptions_blocks_data(self):
        def fail(op,req):
            if op=='describe' and len(self.calls)==2:return httpx.Response(200,json={**self.description,'host':'other.svc.pinecone.io'})
        self.failure=fail
        report=self.run_audit()
        self.assertEqual(report['outcome'],'failed');self.assertEqual(self.calls,['describe','describe'])

    def test_snapshot_tampering_blocks_documents(self):
        path=self.root/'.local/pinecone/snapshots'/(self.snapshot.snapshot_id+'.json')
        value=json.loads(path.read_text());value['records'][0]['text']+='tampered';path.write_text(json.dumps(value))
        report=self.run_audit()
        self.assertEqual(report['reason'],'snapshot');self.assertEqual(self.calls,['describe'])

    def test_confirmation_failure_has_zero_dispatches(self):
        (self.root/'.local/pinecone/target-confirmation.json').write_text('{}')
        report=self.run_audit()
        self.assertEqual(report['reason'],'confirmation');self.assertEqual(self.calls,[])

    def test_fetch_failure_retains_batch_ordinal(self):
        self.failure=lambda op,req:httpx.Response(200,json={'namespace':self.snapshot.namespace,'usage':{'read_units':1},
            'documents':[]}) if op=='fetch' else None
        report=self.run_audit()
        self.assertEqual(report['reason'],'fetch_documents_type');self.assertEqual(report['batch_ordinal'],1)
        self.assertEqual(self.calls,['describe']*2+['list']*2+['fetch'])

    def test_store_record_integrity_rejection_is_sanitized(self):
        def fail(op,req):
            if op!='fetch':return None
            ids=json.loads(req.content)['ids'];values={i:deepcopy(self.store.wire[i]) for i in ids}
            values[ids[0]]['text']=PRIVATE
            return httpx.Response(200,json={'namespace':self.snapshot.namespace,'usage':{'read_units':1},'documents':values})
        self.failure=fail
        report=self.run_audit()
        self.assertEqual(report['reason'],'record_integrity');self.assertTrue(report['publication_preserved'])

    def test_pagination_rejection_is_retained(self):
        self.failure=lambda op,req:httpx.Response(200,json={'namespace':self.snapshot.namespace,'documents':[],
            'usage':{'read_units':1},'pagination':{'next':'repeat'}}) if op=='list' else None
        report=self.run_audit()
        self.assertEqual(report['reason'],'pagination_repeated_token');self.assertEqual(report['page_ordinal'],2)

    def test_report_persistence_failure_does_not_repeat_or_change_publication(self):
        before=[p.read_bytes() for p in self.paths]
        with patch.object(workflow,'corpus_snapshot',return_value=([],workflow.EXPECTED_CORPUS)), \
                patch.object(workflow.os,'fsync',side_effect=OSError(PRIVATE)):
            with self.assertRaises(OSError):self.d.run_diagnostic(self.root,self.settings,
                control_transport=httpx.MockTransport(self.send),documents_transport=httpx.MockTransport(self.send))
        self.assertEqual(before,[p.read_bytes() for p in self.paths]);self.assertEqual(len(self.calls),8)

    def test_report_rejects_unreviewed_data(self):
        report=self.run_audit()
        for name in ('reason','stage','operation','attempts','http_status'):
            with self.subTest(name=name),self.assertRaises((ValueError,TypeError)):
                self.d.save_diagnostic_report(self.root,{**report,name:PRIVATE})
        with self.assertRaises(ValueError):self.d.save_diagnostic_report(self.root,{**report,'raw':PRIVATE})

    def test_report_rejects_fabricated_egress_coverage_and_totals(self):
        report = self.run_audit()
        for changes in ({'egress_complete': True}, {'egress_complete': PRIVATE},
                        {'validated_egress_bytes': 0}, {'responses_with_validated_egress': True},
                        {'validated_egress_bytes': 1, 'responses_with_validated_egress': 7},
                        {'validated_egress_bytes': float('inf'), 'responses_with_validated_egress': 1}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                self.d.save_diagnostic_report(self.root, {**report, **changes})

    def test_source_change_before_or_during_audit_is_rejected(self):
        self.fingerprint='changed-source'
        report=self.run_audit()
        self.assertEqual(report['outcome'],'failed');self.assertEqual(self.calls,['describe'])

    def test_final_source_check_cannot_be_skipped(self):
        def change(op,request):
            if op=='fetch':self.fingerprint='changed-source'
        self.failure=change
        report=self.run_audit()
        self.assertEqual(report['reason'],'source');self.assertEqual(len(self.calls),8)
        self.assertTrue(report['publication_preserved'])

    def test_sibling_metadata_cannot_be_substituted(self):
        def change(op,request):
            if op!='fetch':return None
            ids=json.loads(request.content)['ids'];records={i:deepcopy(self.store.wire[i]) for i in ids}
            records[ids[0]]['required_context_ids']='["unreviewed-sibling"]'
            return httpx.Response(200,json={'namespace':self.snapshot.namespace,'usage':{'read_units':1},'documents':records})
        self.failure=change
        report=self.run_audit()
        self.assertEqual(report['reason'],'record_integrity')

    def test_active_pointer_mismatch_blocks_data(self):
        self.paths[1].write_text(json.dumps({'namespace':'dev-'+'a'*64}))
        report=self.run_audit()
        self.assertEqual(report['outcome'],'failed');self.assertEqual(self.calls,['describe'])

    def test_configuration_failure_still_records_zero_attempts(self):
        from northstar_atlas.vector_store.pinecone_cloud import CloudError
        with patch.object(self.d,'load_settings',side_effect=CloudError('safe configuration failure')):
            report=self.d.run_diagnostic(self.root)
        self.assertEqual(report['outcome'],'failed')
        self.assertEqual(sum(report['attempts'].values()),0)
        self.assertEqual(len(list((self.root/'.local/pinecone/m33').glob('*-diagnostic-audit.json'))),1)

    def test_cli_reporting_failure_is_sanitized_and_never_retried(self):
        from northstar_atlas import pinecone
        out=io.StringIO()
        with patch.object(pinecone,'validate_project_root',return_value=self.root), \
                patch.object(self.d,'run_diagnostic',side_effect=OSError(PRIVATE)) as run, \
                redirect_stdout(out),redirect_stderr(out):
            code=pinecone.main(['diagnostic-audit'])
        self.assertEqual(code,1);self.assertEqual(run.call_count,1)
        self.assertNotIn(PRIVATE,out.getvalue())

    def test_local_request_rejection_does_not_consume_dispatch(self):
        budget=self.d.DispatchBudget();calls=[]
        with self.d.DescribeClient(self.settings,budget,transport=httpx.MockTransport(lambda req:calls.append(req))) as client:
            with patch.object(client.client,'build_request',side_effect=ValueError(PRIVATE)):
                with self.assertRaises(self.d.DiagnosticError):client.describe()
        self.assertEqual(calls,[]);self.assertEqual(sum(budget.counts().values()),0)

    def test_budget_exhaustion_on_first_fetch_reports_refused_operation(self):
        self.page_size=20
        report=self.run_audit()
        self.assertEqual(report['outcome'],'inconclusive')
        self.assertEqual(report['operation'],'fetch')
        self.assertEqual(report['attempts'],{'describe':2,'list':10,'fetch':0})
        self.assertEqual(len(self.calls),12)

    def test_budget_exhaustion_between_fetch_batches(self):
        self.page_size=25
        report=self.run_audit()
        self.assertEqual(report['outcome'],'inconclusive')
        self.assertEqual(report['attempts'],{'describe':2,'list':8,'fetch':2})
        self.assertEqual(len(self.calls),12)

    def test_interrupt_records_inconclusive_without_retry(self):
        def interrupt(op,request):
            if op=='fetch':raise KeyboardInterrupt()
        self.failure=interrupt
        report=self.run_audit()
        self.assertEqual(report['outcome'],'inconclusive');self.assertEqual(report['reason'],'interrupted')
        self.assertEqual(len(self.calls),5);self.assertTrue(report['publication_preserved'])

    def test_external_publication_change_is_detected_without_restore(self):
        def change(op,request):
            if op=='fetch':self.paths[0].write_bytes(self.paths[0].read_bytes()+b' ')
        self.failure=change
        with patch.object(workflow,'corpus_snapshot',return_value=([],workflow.EXPECTED_CORPUS)):
            report=self.d.run_diagnostic(self.root,self.settings,
                control_transport=httpx.MockTransport(self.send),documents_transport=httpx.MockTransport(self.send))
        self.assertEqual(report['outcome'],'failed');self.assertEqual(report['reason'],'publication_changed')
        self.assertFalse(report['publication_preserved'])
        self.assertTrue(self.paths[0].read_bytes().endswith(b'    '))


if __name__ == '__main__': unittest.main()
