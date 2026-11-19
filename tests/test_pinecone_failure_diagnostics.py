"""Sanitized failure evidence; synthetic HTTP and isolated publication fixtures."""

from contextlib import ExitStack, nullcontext, redirect_stderr
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from northstar_atlas.vector_store.pinecone_documents import DocumentsIndex, DocumentsError
from northstar_atlas.vector_store.pinecone_workflow import RequestMeter, save_failure_report
from northstar_atlas.vector_store.records import VectorStoreError

ROOT = Path(__file__).resolve().parents[1]
NS = 'dev-' + 'a' * 64
HOST = 'https://synthetic.svc.eastus2.pinecone.io'
PRIVATE = 'SYNTHETIC_PRIVATE_VALUE_DO_NOT_RECORD'


class FailureDiagnosticsTests(unittest.TestCase):
    def test_search_egress_coverage_and_rejection_preserve_prior_totals(self):
        sequences = (
            ([{'read_units': 2}, {'read_units': 3}], None, 0, False),
            ([{'read_units': 2, 'egress_bytes': 0}, {'read_units': 3, 'egress_bytes': 9}], 9, 2, True),
            ([{'read_units': 2, 'egress_bytes': 7}, {'read_units': 3}], 7, 1, False),
        )
        for rows, total, count, complete in sequences:
            meter = RequestMeter()
            pending = iter(rows + [{'read_units': 100, 'egress_bytes': PRIVATE}])
            calls = []
            def send(request):
                calls.append(request)
                return httpx.Response(200, json={'namespace': NS, 'matches': [], 'usage': next(pending)})
            with DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(send),
                    observe=meter.observe_documents, observe_egress=meter.observe_egress) as index:
                def search():
                    return index.search(namespace=NS, top_k=1,
                        score_by=[{'type':'dense_vector','field':'embedding','values':[1.,0.,0.]}],
                        include_fields=['text'], filter={'record_id':{'$eq':'a'}})
                for _ in rows: search()
                report = meter.report()
                self.assertEqual(report['reported_read_units'], 5)
                self.assertEqual(report['validated_egress_bytes'], total)
                self.assertEqual(report['responses_with_validated_egress'], count)
                self.assertEqual(report['egress_complete'], complete)
                with self.assertRaises(VectorStoreError): search()
                report = meter.report()
                self.assertEqual(report['reported_read_units'], 5)
                self.assertEqual(report['validated_egress_bytes'], total)
                self.assertEqual(report['responses_with_validated_egress'], count)
                self.assertFalse(report['egress_complete'])
                self.assertEqual(len(calls), 3)
                self.assertNotIn(PRIVATE, json.dumps(report))

    def test_fetch_failure_retains_preceding_search_egress(self):
        meter = RequestMeter()
        calls = []
        def send(request):
            calls.append(request)
            if len(calls) == 4:
                return httpx.Response(200, json={'namespace': NS})
            if request.url.path.endswith('/search'):
                return httpx.Response(200, json={'namespace': NS, 'matches': [],
                    'usage': {'read_units': 2, 'egress_bytes': 7}})
            return httpx.Response(200, json={'namespace': NS, 'documents': {},
                'usage': {'read_units': 3, 'egress_bytes': 4}})
        with DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(send),
                observe=meter.observe_documents, observe_egress=meter.observe_egress,
                on_failure=meter.record_documents_failure) as index:
            for _ in range(2):
                index.search(namespace=NS, top_k=1,
                    score_by=[{'type':'dense_vector','field':'embedding','values':[1.,0.,0.]}],
                    include_fields=['text'], filter={'record_id':{'$eq':'a'}})
            index.fetch(namespace=NS, ids=['a'])
            self.assertTrue(meter.report()['egress_complete'])
            # A subsequent rejected fetch must retain all three validated responses.
            with self.assertRaises(VectorStoreError): index.fetch(namespace=NS, ids=['a'])
            data = meter.documents_failure.to_dict()
            self.assertEqual(data['validated_read_units'], 7)
            self.assertEqual(data['validated_egress_bytes'], 18)
            self.assertEqual(data['responses_with_validated_egress'], 3)
            self.assertFalse(data['egress_complete'])
            self.assertEqual(len(calls), 4)

    def test_egress_aggregation_coverage_and_atomic_failure(self):
        sequences = (
            ([{'read_units': 2}, {'read_units': 3}], None, 0, False),
            ([{'read_units': 2, 'egress_bytes': 0}, {'read_units': 3, 'egress_bytes': 9}], 9, 2, True),
            ([{'read_units': 2, 'egress_bytes': 7}, {'read_units': 3}], 7, 1, False),
        )
        for rows, total, count, complete in sequences:
            meter = RequestMeter()
            pending = iter(rows + [{'read_units': 100, 'egress_bytes': PRIVATE}])
            def send(request):
                return httpx.Response(200, json={'namespace': NS, 'documents': {}, 'usage': next(pending)})
            with DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(send),
                    observe=meter.observe_documents, observe_egress=meter.observe_egress,
                    on_failure=meter.record_documents_failure) as index:
                for _ in rows: index.fetch(namespace=NS, ids=['a'])
                report = meter.report()
                self.assertEqual(report['reported_read_units'], 5)
                self.assertEqual(report['validated_egress_bytes'], total)
                self.assertEqual(report['responses_with_validated_egress'], count)
                self.assertEqual(report['egress_complete'], complete)
                with self.assertRaises(VectorStoreError): index.fetch(namespace=NS, ids=['a'])
                data = meter.documents_failure.to_dict()
                self.assertEqual(data['validation_reason'], 'usage_egress_bytes')
                self.assertEqual(data['validated_read_units'], 5)
                self.assertEqual(data['validated_egress_bytes'], total)
                self.assertEqual(data['responses_with_validated_egress'], count)
                self.assertFalse(data['egress_complete'])
                self.assertFalse(data['usage_complete'])
                self.assertNotIn(PRIVATE, json.dumps(data))

    def test_unavailable_documents_usage_remains_unknown_in_workflow_report(self):
        meter = RequestMeter()
        meter.observe_documents('list', None)
        self.assertIsNone(meter.report()['reported_read_units'])
        self.assertIsNone(meter.report()['validated_egress_bytes'])
        self.assertFalse(meter.report()['egress_complete'])

    def failure(self, body, *, fetch=False, status=200):
        meter = RequestMeter(); attempts = []
        def send(request):
            attempts.append(request)
            return httpx.Response(status, content=body if isinstance(body, bytes) else json.dumps(body).encode(),
                                  headers={'location': 'https://untrusted.invalid/' + PRIVATE})
        with DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(send),
                            observe=meter.observe_documents, on_failure=meter.record_documents_failure) as index:
            with self.assertRaises((VectorStoreError, DocumentsError)) as caught:
                if fetch: index.fetch(namespace=NS, ids=['a'])
                else: list(index.list(namespace=NS).pages())
        self.assertEqual(len(attempts), 1)
        data = meter.documents_failure.to_dict()
        encoded = json.dumps(data)
        for private in (PRIVATE, HOST, NS, 'untrusted.invalid'):
            self.assertNotIn(private, encoded)
        self.assertNotIn(PRIVATE, str(caught.exception))
        return data

    def test_rejection_categories_are_distinct_without_widening_acceptance(self):
        base = {'namespace': NS, 'documents': [{'_id': 'a'}], 'usage': {'read_units': 2}}
        cases = [
            (b'not-json-' + PRIVATE.encode(), 'json_malformed', 'json'),
            (b'{"x":1,"x":2}', 'json_duplicate_key', 'json'),
            (b'{"x":NaN}', 'json_nonfinite_number', 'json'),
            ([], 'envelope_type', 'envelope'),
            ({'namespace': NS}, 'envelope_missing_fields', 'envelope'),
            ({**base, PRIVATE: PRIVATE}, 'envelope_extra_fields', 'envelope'),
            ({**base, 'namespace': PRIVATE}, 'namespace_mismatch', 'envelope'),
            ({**base, 'usage': []}, 'usage_type', 'usage'),
            ({**base, 'usage': {}}, 'usage_missing_fields', 'usage'),
            ({**base, 'usage': {'read_units': 2, PRIVATE: PRIVATE}}, 'usage_extra_fields', 'usage'),
            ({**base, 'usage': {'read_units': True}}, 'usage_read_units', 'usage'),
            ({**base, 'documents': {}}, 'document_list_type', 'document'),
            ({**base, 'documents': [{'_id': 'a', PRIVATE: PRIVATE}]}, 'document_list_fields', 'document'),
            ({**base, 'documents': [{'_id': None}]}, 'document_id', 'document'),
            ({**base, 'pagination': {PRIVATE: PRIVATE}}, 'pagination_fields', 'pagination'),
            ({**base, 'pagination': {'next': ''}}, 'pagination_token', 'pagination'),
        ]
        for body, reason, group in cases:
            with self.subTest(reason=reason):
                data = self.failure(body)
                self.assertEqual((data['operation'], data['validation_reason'], data['field_group']), ('list', reason, group))
                self.assertEqual(data['http_status'], 200)
                self.assertEqual(data['page_ordinal'], 1)
                self.assertIsNone(data['batch_ordinal'])
                self.assertEqual(data['http_transport_attempts'], {'list': 1, 'fetch': 0, 'search': 0})

    def test_fetch_specific_rejections_retain_batch_and_validated_usage(self):
        base = {'namespace': NS, 'documents': {'a': {'_id': 'a'}}, 'usage': {'read_units': 3}}
        cases = [({**base, 'pagination': {'next': PRIVATE}}, 'fetch_continuation'),
                 ({**base, 'documents': []}, 'fetch_documents_type'),
                 ({**base, 'documents': {'other': {'_id': 'other'}}}, 'fetch_unexpected_ids'),
                 ({**base, 'documents': {'a': {'_id': PRIVATE}}}, 'fetch_record_identity')]
        for body, reason in cases:
            with self.subTest(reason=reason):
                data = self.failure(body, fetch=True)
                self.assertEqual(data['validation_reason'], reason)
                self.assertEqual((data['operation'], data['batch_ordinal'], data['page_ordinal']), ('fetch', 1, None))
                self.assertEqual(data['validated_read_units'], 3)
                self.assertTrue(data['usage_complete'])

    def test_unknown_usage_is_not_reported_as_zero_charges(self):
        data = self.failure({'namespace': NS, 'documents': [], 'usage': {'read_units': '2'}})
        self.assertIsNone(data['validated_read_units'])
        self.assertEqual(data['responses_with_validated_usage'], 0)
        self.assertFalse(data['usage_complete'])

    def test_page_two_and_fetch_two_have_actual_transport_counts(self):
        for operation in ('list', 'fetch'):
            meter = RequestMeter(); calls = []
            def send(request):
                calls.append(request)
                body = {'namespace': NS, 'documents': [] if operation == 'list' else {}, 'usage': {'read_units': 2}}
                if operation == 'list': body['pagination'] = {'next': 'synthetic-next'}
                if len(calls) == 2: body['usage'][PRIVATE] = PRIVATE
                return httpx.Response(200, json=body)
            with DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(send),
                                on_failure=meter.record_documents_failure) as index:
                with self.assertRaises(VectorStoreError):
                    if operation == 'list': list(index.list(namespace=NS).pages())
                    else:
                        index.fetch(namespace=NS, ids=['a']); index.fetch(namespace=NS, ids=['b'])
            data = meter.documents_failure.to_dict()
            self.assertEqual(data['page_ordinal' if operation == 'list' else 'batch_ordinal'], 2)
            self.assertEqual(data['http_transport_attempts'][operation], 2)
            self.assertEqual(data['validated_read_units'], 2)
            self.assertFalse(data['usage_complete'])

    def test_timeout_http_error_and_redirect_report_without_replay(self):
        for status in (302, 401, 403, 408, 429, 500, 502, 503, 504):
            data = self.failure(PRIVATE.encode(), status=status)
            self.assertEqual((data['http_status'], data['validation_reason']), (status, 'http_rejection'))
        for kind in (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            meter = RequestMeter(); calls = []
            def send(request):
                calls.append(request); raise kind(PRIVATE, request=request)
            with DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(send),
                                on_failure=meter.record_documents_failure) as index:
                with self.assertRaises(DocumentsError): index.fetch(namespace=NS, ids=['a'])
            data = meter.documents_failure.to_dict()
            self.assertEqual(len(calls), 1)
            self.assertEqual(data['validation_reason'], 'transport_timeout')
            self.assertIsNone(data['http_status'])
            self.assertNotIn(PRIVATE, json.dumps(data))

    def test_serialization_error_does_not_inflate_transport_attempts(self):
        meter = RequestMeter(); calls = []
        with DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(lambda r: calls.append(r)),
                            on_failure=meter.record_documents_failure) as index:
            with patch.object(index._client, 'post', side_effect=ValueError(PRIVATE)):
                with self.assertRaises(DocumentsError): index.fetch(namespace=NS, ids=['a'])
        self.assertEqual(calls, [])
        self.assertEqual(meter.documents_failure.to_dict()['http_transport_attempts']['fetch'], 0)


class FailureReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.local/tmp', prefix='diagnostic-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'pyproject.toml').write_text('[project]\nname="northstar-atlas"\n')

    def diagnostic(self):
        meter = RequestMeter()
        with DocumentsIndex(host=HOST, api_key=PRIVATE,
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'namespace': NS})),
                on_failure=meter.record_documents_failure) as index:
            with self.assertRaises(VectorStoreError): list(index.list(namespace=NS).pages())
        return meter.documents_failure

    def test_private_report_contains_only_validated_fields(self):
        path = save_failure_report(self.root, 'search', 'startup_audit', self.diagnostic())
        self.assertEqual(path.parent, self.root / '.local/pinecone/m33')
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        value = json.loads(path.read_text())
        self.assertEqual(set(value), {'schema_version', 'result', 'command', 'stage', 'diagnostic'})
        self.assertEqual(value['stage'], 'startup_audit')
        self.assertEqual(value['diagnostic']['operation'], 'list')
        for secret in (PRIVATE, NS, HOST): self.assertNotIn(secret, path.read_text())

    def test_tampered_diagnostics_are_rejected_before_writing(self):
        original = self.diagnostic()
        for changes in ({'operation': PRIVATE}, {'validation_reason': PRIVATE}, {'http_status': PRIVATE},
                        {'ordinal': True}, {'list_attempts': -1}, {'validated_read_units': float('nan')}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                save_failure_report(self.root, 'search', 'startup_audit', replace(original, **changes))
        self.assertFalse((self.root / '.local').exists())

    def test_egress_diagnostic_tampering_is_rejected_before_persistence(self):
        original = self.diagnostic()
        for changes in ({'validated_egress_bytes': PRIVATE}, {'validated_egress_bytes': 0},
                        {'responses_with_validated_egress': True}, {'responses_with_validated_egress': 1},
                        {'validated_egress_bytes': -1, 'responses_with_validated_egress': 1}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                save_failure_report(self.root, 'search', 'startup_audit', replace(original, **changes))
        self.assertFalse((self.root / '.local').exists())

    def test_symlink_path_is_rejected_without_touching_destination(self):
        destination = self.root / 'elsewhere'; destination.mkdir()
        (self.root / '.local').symlink_to(destination, target_is_directory=True)
        with self.assertRaises(ValueError):
            save_failure_report(self.root, 'search', 'startup_audit', self.diagnostic())
        self.assertEqual(list(destination.iterdir()), [])

    def test_failed_report_sync_is_not_retried_or_reported_as_success(self):
        diagnostic = self.diagnostic()
        with patch('northstar_atlas.vector_store.pinecone_workflow.os.fsync', side_effect=OSError(PRIVATE)) as sync:
            with self.assertRaises(OSError):
                save_failure_report(self.root, 'search', 'startup_audit', diagnostic)
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(len(list((self.root / '.local/pinecone/m33').glob('*.json'))), 1)

    def test_filename_collision_never_overwrites_existing_report(self):
        from datetime import datetime, timezone
        with patch('northstar_atlas.vector_store.pinecone_workflow.datetime') as clock:
            clock.now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
            first = save_failure_report(self.root, 'search', 'startup_audit', self.diagnostic())
            before = first.read_bytes()
            with self.assertRaises(FileExistsError):
                save_failure_report(self.root, 'search', 'startup_audit', self.diagnostic())
            self.assertEqual(first.read_bytes(), before)


class AuditCLITests(unittest.TestCase):
    def setUp(self):
        from test_pinecone_store import PineconeTests
        from test_pinecone_cloud import cloud_settings
        PineconeTests.setUpClass()
        self.f = PineconeTests(); self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.settings = cloud_settings()
        self.f.target = replace(self.f.target, host=HOST, project_id=self.settings.project_id)
        self.f.description['host'] = HOST
        self.f.documents._host = HOST
        self.f.store = self.f.make_store()
        self.f.store.synchronize()  # Existing SDK publication path, synthetic HTTP only.

    def run_cli(self, *, persistence_failure=False, invalidation_failure=False, incomplete=False):
        from northstar_atlas import pinecone
        f = self.f
        pointer_before = (f.store.state.path / 'active.json').read_bytes()
        if incomplete:
            with f.store.state.lock(): f.store.state.write_run(f.store.namespace, f.store._manifest('incomplete', phase='failed audit'))
        manifest_before = f.store.state.read_run(f.store.namespace)
        calls=[]
        def connect(client, meter, settings, *, read_only):
            self.assertTrue(read_only)
            def send(request):
                calls.append(request)
                return httpx.Response(200, json={'namespace': f.store.namespace, 'documents': [],
                    'usage': {'read_units': 1, PRIVATE: PRIVATE}})
            index = DocumentsIndex(host=HOST, api_key=PRIVATE, transport=httpx.MockTransport(send),
                observe=meter.observe_documents, on_failure=meter.record_documents_failure)
            f.documents=index.documents
            f.store=f.make_store(f.store.snapshot)
            if invalidation_failure:
                f.store.state.write_run=Mock(side_effect=OSError(PRIVATE))
            return f.target,index
        output=io.StringIO()
        with ExitStack() as stack:
            for name,value in (('validate_project_root', f.root),('load_settings',self.settings),('check_confirmation',None),
                               ('active_snapshot',f.store.snapshot)):
                stack.enter_context(patch.object(pinecone,name,return_value=value))
            stack.enter_context(patch.object(pinecone,'sdk_client',return_value=nullcontext(Mock())))
            stack.enter_context(patch.object(pinecone.RequestMeter,'attach'))
            stack.enter_context(patch.object(pinecone,'connect_target',side_effect=connect))
            stack.enter_context(patch.object(pinecone,'open_store',side_effect=lambda *a: f.store))
            model=stack.enter_context(patch.object(pinecone,'MiniLMProvider',side_effect=AssertionError('must not initialize model')))
            if persistence_failure:
                stack.enter_context(patch.object(pinecone,'save_failure_report',side_effect=OSError(PRIVATE)))
            with redirect_stderr(output):
                result=pinecone.main(['search','synthetic question'])
        self.assertEqual(result,1)
        model.assert_not_called()
        self.assertEqual((f.store.state.path / 'active.json').read_bytes(), pointer_before)
        self.assertNotIn(PRIVATE,output.getvalue())
        reports=list((f.root / '.local/pinecone/m33').glob('*-failure.json'))
        if incomplete or invalidation_failure:
            self.assertEqual(f.store.state.read_run(f.store.namespace),manifest_before)
        else:
            self.assertEqual(f.store.state.read_run(f.store.namespace)['state'],'incomplete')
        return calls,reports,output.getvalue()

    def test_failure_report_survives_audit_invalidation_and_cli_exit(self):
        calls,reports,output=self.run_cli()
        self.assertEqual(len(calls),1)
        self.assertEqual(len(reports),1)
        value=json.loads(reports[0].read_text())
        self.assertEqual(value['stage'],'startup_audit')
        self.assertEqual(value['diagnostic']['validation_reason'],'usage_extra_fields')
        self.assertEqual(value['diagnostic']['http_transport_attempts'],{'list':1,'fetch':0,'search':0})
        self.assertEqual(reports[0].stat().st_mode & 0o777,0o600)

    def test_report_write_failure_does_not_retry_or_restore_publication(self):
        calls,reports,output=self.run_cli(persistence_failure=True)
        self.assertEqual(len(calls),1)
        self.assertEqual(reports,[])
        self.assertIn('report unavailable',output)

    def test_invalidation_failure_does_not_erase_diagnostic(self):
        calls,reports,output=self.run_cli(invalidation_failure=True)
        self.assertEqual(len(calls),1)
        self.assertEqual(len(reports),1)
        self.assertEqual(json.loads(reports[0].read_text())['diagnostic']['validation_reason'],'usage_extra_fields')

    def test_already_incomplete_publication_stays_blocked_without_transport(self):
        calls,reports,output=self.run_cli(incomplete=True)
        self.assertEqual(calls,[])
        self.assertEqual(reports,[])


if __name__ == '__main__': unittest.main()
