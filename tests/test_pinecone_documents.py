"""Read-only Documents REST contract; synthetic transports, never cloud access."""

from copy import deepcopy
import json
import logging
import unittest
from unittest.mock import patch

import httpx

from northstar_atlas.vector_store.pinecone_documents import DocumentsIndex, DocumentsError
from northstar_atlas.vector_store.records import VectorStoreError

HOST = 'https://synthetic.svc.eastus2.pinecone.io'
NAMESPACE = 'dev-' + 'a' * 64
KEY = 'synthetic-test-only-key'


class DocumentsTests(unittest.TestCase):
    def test_list_fetch_documented_usage_and_integer_boundaries(self):
        for usage in ({'read_units': 0}, {'read_units': 3, 'egress_bytes': 0},
                      {'read_units': 2147483647, 'egress_bytes': 9223372036854775807}):
            for operation in ('list', 'fetch'):
                with self.subTest(usage=usage, operation=operation):
                    def send(request):
                        return httpx.Response(200, json={'namespace': NAMESPACE, 'usage': usage,
                            'documents': [] if operation == 'list' else {}})
                    documents = self.client(send)
                    if operation == 'list': list(documents.list(namespace=NAMESPACE).pages())
                    else: documents.fetch(namespace=NAMESPACE, ids=['a'])
                    self.assertEqual(documents._read_units, usage['read_units'])

    def test_invalid_usage_is_atomic_and_single_attempt(self):
        invalid = [{}, {'egress_bytes': 0}, {'read_units': 1, 'readUnits': 2},
                   {'read_units': 1, 'egressBytes': 2}, {'read_units': 1, 'unknown': KEY}]
        for field, maximum in (('read_units', 2147483647), ('egress_bytes', 9223372036854775807)):
            invalid += [dict({'read_units': 1}, **{field: value})
                        for value in (True, 1.0, '1', None, -1, maximum + 1)]
        for usage in invalid:
            for operation in ('list', 'fetch'):
                calls = []
                def send(request):
                    calls.append(request)
                    return httpx.Response(200, json={'namespace': NAMESPACE, 'usage': usage,
                        'documents': [] if operation == 'list' else {}})
                documents = self.client(send)
                with self.subTest(usage=usage, operation=operation), self.assertRaises(VectorStoreError):
                    if operation == 'list': list(documents.list(namespace=NAMESPACE).pages())
                    else: documents.fetch(namespace=NAMESPACE, ids=['a'])
                self.assertEqual(len(calls), 1)
                self.assertEqual((documents._read_units, documents._usage_responses), (0, 0))

    def test_usage_nonfinite_and_duplicate_json_rejected(self):
        usages = ['{"read_units":1,"read_units":2}', '{"read_units":1,"egress_bytes":0,"egress_bytes":1}']
        usages += ['{"read_units":1,"egress_bytes":' + token + '}'
                   for token in ('NaN', 'Infinity', '-Infinity', '1e999')]
        for usage in usages:
            body = ('{"namespace":"' + NAMESPACE + '","documents":[],"usage":' + usage + '}').encode()
            documents = self.client(lambda request: httpx.Response(200, content=body))
            with self.subTest(usage=usage), self.assertRaises(VectorStoreError):
                list(documents.list(namespace=NAMESPACE).pages())
            self.assertEqual(documents._usage_responses, 0)

    def test_search_usage_accepts_required_and_optional_integer_boundaries(self):
        for usage in ({'read_units': 0}, {'read_units': 1, 'egress_bytes': 0},
                      {'read_units': 2147483647, 'egress_bytes': 9223372036854775807}):
            calls = []
            def send(request):
                calls.append(request)
                return httpx.Response(200, json={'namespace': NAMESPACE, 'matches': [
                    {'_id': 'a', '_score': .75, 'text': 'evidence'}], 'usage': usage})
            documents = self.client(send)
            result = self.operations(documents)[2]()
            self.assertEqual(result.matches[0].to_dict(), {'_id': 'a', '_score': .75, 'text': 'evidence'})
            self.assertEqual(len(calls), 1)
            body = json.loads(calls[0].content)
            self.assertEqual(body['filter'], {'status': {'$eq': 'approved'}})
            self.assertEqual(body['score_by'], [{'type': 'dense_vector', 'field': 'embedding', 'values': [1., 0., 0.]}])
            self.assertEqual(documents._read_units, usage['read_units'])

    def test_search_invalid_usage_is_atomic_without_retry(self):
        invalid = [{}, {'egress_bytes': 1}, {'read_units': 1, 'embed_total_tokens': 0},
                   {'read_units': 1, 'embed_total_tokens': 12}, {'read_units': 1, 'unknown': KEY},
                   {'read_units': 1, 'readUnits': 1}, {'read_units': 1, 'egressBytes': 1}]
        for field, maximum in (('read_units', 2147483647), ('egress_bytes', 9223372036854775807)):
            invalid += [dict({'read_units': 1}, **{field: v}) for v in (True, 1., '1', None, -1, maximum+1)]
        for usage in invalid:
            calls = []
            def send(request):
                calls.append(request)
                return httpx.Response(200, json={'namespace': NAMESPACE, 'matches': [], 'usage': usage})
            documents = self.client(send)
            with self.subTest(usage=usage), self.assertRaises(VectorStoreError) as caught:
                self.operations(documents)[2]()
            self.assertNotIn(KEY, str(caught.exception))
            self.assertEqual(len(calls), 1)
            self.assertEqual((documents._read_units, documents._usage_responses,
                              documents._egress_bytes, documents._egress_responses), (0, 0, 0, 0))

    def test_search_nonfinite_duplicate_and_malformed_json_is_rejected(self):
        usages = ['{"read_units":1,"read_units":2}', '{"read_units":1,"egress_bytes":0,"egress_bytes":1}']
        usages += ['{"read_units":1,"egress_bytes":'+v+'}' for v in ('NaN','Infinity','-Infinity','1e999')]
        bodies = [(' {"namespace":"'+NAMESPACE+'","matches":[],"usage":'+u+'}').encode() for u in usages]
        bodies += [b'not-json']
        for body in bodies:
            documents = self.client(lambda request:httpx.Response(200, content=body))
            with self.subTest(body=body), self.assertRaises(VectorStoreError):self.operations(documents)[2]()
            self.assertEqual(documents._usage_responses, 0)

    def client(self, handler):
        client = DocumentsIndex(host=HOST, api_key=KEY, transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return client.documents

    def operations(self, documents):
        return (
            lambda: list(documents.list(namespace=NAMESPACE, limit=100, timeout=10).pages()),
            lambda: documents.fetch(namespace=NAMESPACE, ids=['a'], timeout=10),
            lambda: documents.search(namespace=NAMESPACE, top_k=5, timeout=10,
                score_by=[{'type': 'dense_vector', 'field': 'embedding', 'values': [1., 0., 0.]}],
                include_fields=['text'], filter={'status': {'$eq': 'approved'}}),
        )

    def test_requests_are_exact_single_attempt_and_preserve_fields(self):
        calls = []
        def send(request):
            calls.append(request)
            operation = request.url.path.rsplit('/', 1)[-1]
            bodies = {'list': {'documents': [{'_id': 'a'}]},
                      'fetch': {'documents': {'a': {'_id': 'a', 'text': 'evidence', 'embedding': [1., 0., 0.]}}},
                      'search': {'matches': [{'_id': 'a', '_score': .75, 'text': 'evidence'}]}}
            return httpx.Response(200, json={'namespace': NAMESPACE, 'usage': {'read_units': 1}, **bodies[operation]})
        documents = self.client(send)
        outputs = [operation() for operation in self.operations(documents)]
        self.assertEqual(len(calls), 3)
        for request in calls:
            self.assertEqual(request.method, 'POST')
            self.assertEqual(str(request.url).split('/namespaces/')[0], HOST)
            self.assertEqual(request.headers['X-Pinecone-Api-Version'], '2026-07')
            self.assertEqual(request.headers['Api-Key'], KEY)
            self.assertEqual(request.extensions['timeout'], {'connect': 5., 'read': 10., 'write': 10., 'pool': 5.})
        search = json.loads(calls[-1].content)
        self.assertEqual(search['score_by'], [{'type': 'dense_vector', 'field': 'embedding', 'values': [1., 0., 0.]}])
        self.assertEqual(search['filter'], {'status': {'$eq': 'approved'}})
        self.assertEqual(outputs[1].documents['a'].to_dict()['embedding'], [1., 0., 0.])
        self.assertEqual(outputs[2].matches[0].to_dict()['_score'], .75)
        for name in ('upsert', 'update', 'delete', 'delete_namespace'):
            self.assertFalse(hasattr(documents, name))

    def test_transport_failures_never_replay_and_are_sanitized(self):
        errors = (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
                  httpx.PoolTimeout, httpx.ConnectError, httpx.ReadError)
        for error in errors:
            for operation_number in range(3):
                calls = []
                def send(request):
                    calls.append(request)
                    raise error(f'{KEY} {HOST} private payload', request=request)
                documents = self.client(send)
                with self.subTest(error=error, operation=operation_number):
                    with self.assertRaises(DocumentsError) as caught:
                        self.operations(documents)[operation_number]()
                    self.assertEqual(len(calls), 1)
                    for secret in (KEY, HOST, 'private payload'):
                        self.assertNotIn(secret, str(caught.exception))
                    self.assertTrue(caught.exception.__suppress_context__)

    def test_http_errors_and_redirects_never_replay(self):
        for status in (301, 302, 307, 308, 401, 403, 408, 429, 500, 502, 503, 504):
            for number in range(3):
                calls = []
                def send(request):
                    calls.append(request)
                    return httpx.Response(status, text=KEY, headers={'location': 'https://other.invalid/'})
                with self.subTest(status=status, operation=number):
                    with self.assertRaises(DocumentsError):
                        self.operations(self.client(send))[number]()
                    self.assertEqual(len(calls), 1)

    def test_invalid_host_and_namespace_fail_before_transport(self):
        for host in ('http://a.pinecone.io', 'https://a.pinecone.io.evil.invalid',
                     'https://user@a.pinecone.io', HOST + '/path', HOST + '?secret=x',
                     HOST + '#fragment', HOST + ':443', 'https://127.0.0.1'):
            with self.subTest(host=host), self.assertRaises(DocumentsError):
                DocumentsIndex(host=host, api_key=KEY)
        calls = []
        documents = self.client(lambda request: calls.append(request))
        for namespace in ('', '../other', NAMESPACE + '/', 'another'):
            with self.subTest(namespace=namespace), self.assertRaises(DocumentsError):
                documents.fetch(namespace=namespace, ids=['a'])
        self.assertEqual(calls, [])

    def test_malformed_envelopes_fail_closed(self):
        valid = {'namespace': NAMESPACE, 'documents': {'a': {'_id': 'a'}}, 'usage': {'read_units': 1}}
        bodies = [b'not json', b'[]', b'{"namespace":1,"namespace":2}', b'{"x":NaN}']
        for field in valid:
            value = deepcopy(valid); del value[field]; bodies.append(json.dumps(value).encode())
        for extra in ({'unexpected': KEY}, {'namespace': 'wrong'}, {'usage': {'read_units': True}},
                      {'usage': {'read_units': -1}}, {'documents': []},
                      {'pagination': {'next': 'more'}}):
            bodies.append(json.dumps({**valid, **extra}).encode())
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(VectorStoreError):
                self.client(lambda request: httpx.Response(200, content=body)).fetch(namespace=NAMESPACE, ids=['a'])

    def test_pagination_terminates_and_repeated_token_is_rejected(self):
        calls = []
        def send(request):
            body = json.loads(request.content); calls.append(body)
            value = {'namespace': NAMESPACE, 'documents': [{'_id': str(len(calls))}], 'usage': {'read_units': 1}}
            if len(calls) == 1: value['pagination'] = {'next': 'next-page'}
            return httpx.Response(200, json=value)
        pages = list(self.client(send).list(namespace=NAMESPACE).pages())
        self.assertEqual([p.items[0].id for p in pages], ['1', '2'])
        self.assertEqual(calls[1]['pagination_token'], 'next-page')
        calls.clear()
        def repeated(request):
            calls.append(request)
            return httpx.Response(200, json={'namespace': NAMESPACE, 'documents': [],
                'usage': {'read_units': 1}, 'pagination': {'next': 'repeat'}})
        with self.assertRaises(VectorStoreError):
            list(self.client(repeated).list(namespace=NAMESPACE).pages())
        self.assertEqual(len(calls), 2)

    def test_omitted_next_is_valid_final_page_and_fetch_is_complete(self):
        for pagination in ({}, {'next': None}):
            calls = []
            def send(request):
                calls.append(request)
                records = [] if request.url.path.endswith('/list') else {}
                return httpx.Response(200, json={'namespace': NAMESPACE, 'documents': records,
                    'usage': {'read_units': 1}, 'pagination': pagination})
            documents = self.client(send)
            self.assertEqual(len(list(documents.list(namespace=NAMESPACE).pages())), 1)
            self.assertEqual(documents.fetch(namespace=NAMESPACE, ids=['a']).documents, {})
            self.assertEqual(len(calls), 2)

    def test_usage_observer_and_logging_do_not_expose_private_data(self):
        observed = []
        def send(request):
            logging.getLogger('httpx').info('%s %s payload', KEY, HOST)
            return httpx.Response(200, json={'namespace': NAMESPACE, 'documents': {}, 'usage': {'read_units': 3}})
        with DocumentsIndex(host=HOST, api_key=KEY, transport=httpx.MockTransport(send),
                            observe=lambda operation, units: observed.append((operation, units))) as index:
            with self.assertNoLogs('httpx', level='DEBUG'):
                index.documents.fetch(namespace=NAMESPACE, ids=['a'])
        self.assertEqual(observed, [('fetch', None), ('fetch', 3)])

    def test_real_http_transport_connect_attempt_not_retried(self):
        import httpcore
        with DocumentsIndex(host=HOST, api_key=KEY) as index:
            with patch('httpcore._backends.sync.SyncBackend.connect_tcp',
                       side_effect=httpcore.ConnectError('synthetic')) as connect:
                with self.assertRaises(DocumentsError):
                    index.documents.fetch(namespace=NAMESPACE, ids=['a'])
            self.assertEqual(connect.call_count, 1)


class StoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_pinecone_store import PineconeTests
        from dataclasses import replace
        PineconeTests.setUpClass()
        self.fixture = PineconeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.target = replace(f.target, host=HOST)
        f.description['host'] = HOST
        f.documents._host = HOST
        f.store = f.make_store()
        self.calls = []

    def read_only(self):
        from northstar_atlas.vector_store.pinecone_workflow import RequestMeter
        f = self.fixture
        def send(request):
            body = json.loads(request.content)
            self.calls.append((request.url.path.rsplit('/', 1)[-1], body))
            if request.url.path.endswith('/list'):
                ids = sorted(f.http.spaces[f.store.namespace])
                start = int(body.get('pagination_token', '0'))
                value = {'namespace': f.store.namespace, 'usage': {'read_units': 1},
                         'documents': [{'_id': i} for i in ids[start:start + body['limit']]]}
                if start + body['limit'] < len(ids):
                    value['pagination'] = {'next': str(start + body['limit'])}
                return httpx.Response(200, json=value)
            return f.http.post(request.url.path, json=body, timeout=10)
        self.meter = RequestMeter()
        index = DocumentsIndex(host=HOST, api_key=KEY, transport=httpx.MockTransport(send), observe=self.meter.observe_documents)
        self.addCleanup(index.close)
        f.documents = index.documents
        f.store = f.make_store(f.store.snapshot)

    def test_full_189_record_audit_batches_precision_and_publication_unchanged(self):
        from dataclasses import replace
        from test_vector_store import FakeProvider
        from northstar_atlas.embeddings import embed_chunks
        from northstar_atlas.vector_store.preparation import freeze_snapshot
        from northstar_atlas.vector_store.records import IndexConfig
        class WideProvider(FakeProvider):
            model = replace(FakeProvider.model, dimension=384)

            def embed_documents(self, texts):
                return [vector + [0.] * 381 for vector in super().embed_documents(texts)]
        f = self.fixture
        f.provider = WideProvider()
        f.config = IndexConfig(f.provider.model, 256)
        f.description['schema']['fields']['embedding']['dimension'] = 384
        chunks = [replace(f.hr, chunk_id=f'synthetic-{i:03}') for i in range(189)]
        snapshot = freeze_snapshot(embed_chunks(chunks, f.provider), f.config, f.current)
        f.store = f.make_store(snapshot)
        f.store.synchronize()  # Original SDK writing path, entirely in-memory.
        before = {p.name: p.read_bytes() for p in f.store.state.path.glob('*.json')}
        self.read_only()
        f.store.audit()
        self.assertEqual([name for name, _ in self.calls], ['list', 'list', 'fetch', 'fetch', 'fetch', 'fetch'])
        self.assertEqual([len(b['ids']) for op, b in self.calls if op == 'fetch'], [50, 50, 50, 39])
        self.assertEqual(self.meter.report()['reported_read_units'], 6)
        self.assertEqual(before, {p.name: p.read_bytes() for p in f.store.state.path.glob('*.json')})
        self.assertEqual(f.store.snapshot.records, snapshot.records)
        self.assertFalse(any(op in ('upsert', 'delete', 'update') for op, _ in self.calls))

    def test_status_audit_also_selects_read_only_transport(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from unittest.mock import Mock
        from test_pinecone_cloud import cloud_settings
        from northstar_atlas.vector_store import pinecone_workflow as workflow
        f = self.fixture
        settings = cloud_settings()
        f.target = replace(f.target, project_id=settings.project_id)
        f.store = f.make_store()
        f.store.synchronize(); self.read_only()
        client = SimpleNamespace(indexes=SimpleNamespace(describe=lambda **_: f.description), index=Mock())
        with patch.object(workflow, 'preflight', return_value={'description': f.description}), \
             patch.object(workflow, 'active_snapshot', return_value=f.store.snapshot), \
             patch.object(workflow, 'validate_m33'), \
             patch.object(workflow, 'corpus_snapshot', return_value=([], f.current)), \
             patch.object(workflow, 'open_documents_index', return_value=f.documents) as selected:
            report = workflow.status_report(f.root, settings, client, self.meter)
        self.assertTrue(report['remote_audited'])
        self.assertEqual(selected.call_args.kwargs, {'read_only': True})
        client.index.assert_not_called()

    def test_eligibility_and_unexpected_record_fields_remain_enforced(self):
        f = self.fixture
        f.store.synchronize(); self.read_only()
        self.assertEqual([h.chunk.document_id for h in f.search()], ['HR-001'])
        self.assertEqual(len(f.search(mode='draft_review')), 2)
        body = next(b for op, b in self.calls if op == 'search')
        self.assertIn({'status': {'$eq': 'Approved (Synthetic)'}}, body['filter']['$and'])
        f.http.spaces[f.store.namespace][f.hr.chunk_id]['unexpected'] = 'not reviewed'
        f.store = f.make_store(f.store.snapshot)
        with self.assertRaises(VectorStoreError):
            f.store.audit()
        self.assertEqual(f.store.status()['state'], 'incomplete')

    def test_sibling_fetch_source_identity_and_fail_closed_behavior(self):
        from northstar_atlas.ingestion.chunker import chunk_document
        from northstar_atlas.ingestion.refiner import refine_chunks
        from northstar_atlas.embeddings import embed_chunks
        from northstar_atlas.vector_store.preparation import freeze_snapshot
        from northstar_atlas.vector_store.records import IndexConfig
        f = self.fixture
        f.provider.max_input_tokens = 200
        family = [c for c in chunk_document(next(d for d in f.docs if d.document_id == 'PLAT-001')).chunks if c.section_id == '6.1']
        refined = refine_chunks(family, f.provider).require_complete()
        snapshot = freeze_snapshot(embed_chunks(refined, f.provider), IndexConfig(f.provider.model, 200), f.current)
        f.store = f.make_store(snapshot)
        f.store.synchronize(); self.read_only()
        hit = f.search(mode='draft_review', top_k=1)[0]
        self.assertEqual(len(hit.context), len(refined) - 1)
        self.assertEqual(self.calls[-1][0], 'fetch')
        self.assertTrue(all(c.section_id == '6.1' for c in hit.context))
        del f.http.spaces[f.store.namespace][hit.context[0].chunk_id]
        searches = sum(op == 'search' for op, _ in self.calls)
        with self.assertRaises(VectorStoreError):
            f.search(mode='draft_review', top_k=1)
        self.assertEqual(sum(op == 'search' for op, _ in self.calls), searches + 1)
        self.assertEqual(f.store.status()['state'], 'incomplete')

    def test_target_change_rejected_before_documents_io(self):
        f = self.fixture
        f.store.synchronize(); self.read_only()
        f.description['host'] = 'https://other.pinecone.io'
        with self.assertRaises(VectorStoreError):
            f.store.audit()
        self.assertEqual(self.calls, [])

    def test_source_and_vector_changes_are_not_normalized_away(self):
        for field, value in (('text', 'altered'), ('embedding', [1., 0., 0.]),
                             ('content_hash', 'a' * 64)):
            with self.subTest(field=field):
                # A separate synthetic publication for each integrity failure.
                f = self.fixture
                if field == 'text':
                    f.store.synchronize(); self.read_only()
                original = deepcopy(f.http.spaces[f.store.namespace])
                f.http.spaces[f.store.namespace][f.hr.chunk_id][field] = value
                with self.assertRaises(VectorStoreError):
                    f.store._audit()
                f.http.spaces[f.store.namespace] = original


class SelectionTests(unittest.TestCase):
    def test_cli_search_uses_read_only_and_index_keeps_sdk_selection(self):
        from contextlib import ExitStack, nullcontext, redirect_stdout
        import io
        from unittest.mock import Mock
        from test_pinecone_cloud import cloud_settings, description
        from test_vector_store import ROOT, AS_OF, FakeProvider
        from northstar_atlas import pinecone
        from northstar_atlas.embeddings import embed_chunks
        from northstar_atlas.ingestion.chunker import chunk_document
        from northstar_atlas.ingestion.document_loader import load_documents
        from northstar_atlas.vector_store.preparation import freeze_snapshot
        from northstar_atlas.vector_store.records import IndexConfig
        from northstar_atlas.vector_store.pinecone_store import PineconeTarget, PublicationReport
        provider = FakeProvider()
        chunk = chunk_document(load_documents(ROOT, reference_date=AS_OF)[0]).chunks[0]
        snapshot = freeze_snapshot(embed_chunks([chunk], provider), IndexConfig(provider.model, 256), 'synthetic')
        settings = cloud_settings()
        target = PineconeTarget(settings.name, description()['host'], settings.project_id)
        for command in ('search', 'index'):
            store = Mock()
            store.status.return_value = {'state': 'ready'}
            store.synchronize.return_value = PublicationReport(snapshot.namespace, 1, 0, True)
            with ExitStack() as stack:
                for name, value in (('validate_project_root', ROOT), ('load_settings', settings),
                                    ('check_confirmation', None), ('active_snapshot', snapshot),
                                    ('open_store', store), ('MiniLMProvider', Mock(load_seconds=0)),
                                    ('search_report', {'hits': []}),
                                    ('save_report', ROOT / '.local/synthetic-report.json')):
                    stack.enter_context(patch.object(pinecone, name, return_value=value))
                stack.enter_context(patch.object(pinecone, 'sdk_client', return_value=nullcontext(Mock())))
                stack.enter_context(patch.object(pinecone.RequestMeter, 'attach'))
                selected = stack.enter_context(patch.object(pinecone, 'connect_target',
                                                return_value=(target, nullcontext(Mock()))))
                args = ['search', 'synthetic question', '--reference-date', AS_OF.isoformat()] if command == 'search' else ['index', '--namespace', snapshot.namespace]
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(pinecone.main(args), 0)
                self.assertEqual(selected.call_args.kwargs, {'read_only': command == 'search'})
                if command == 'search':
                    store.audit.assert_called_once_with()
                    store.synchronize.assert_not_called()
                else:
                    store.synchronize.assert_called_once_with()

    def test_shared_qa_retriever_selects_read_only_before_audit(self):
        from contextlib import ExitStack, nullcontext
        from unittest.mock import Mock
        from test_pinecone_cloud import cloud_settings
        from test_vector_store import ROOT
        from northstar_atlas.qa import open_retriever
        from northstar_atlas.vector_store import pinecone_workflow as workflow, pinecone_cloud as cloud
        with ExitStack() as stack:
            stack.enter_context(patch('northstar_atlas.qa.validate_project_root', return_value=ROOT))
            stack.enter_context(patch('northstar_atlas.local_embeddings.MiniLMProvider'))
            stack.enter_context(patch('northstar_atlas.vector_store.corpus_snapshot', return_value=([], 'synthetic')))
            stack.enter_context(patch.object(cloud, 'load_settings', return_value=cloud_settings()))
            stack.enter_context(patch.object(cloud, 'check_confirmation'))
            stack.enter_context(patch.object(cloud, 'sdk_client', return_value=nullcontext(Mock())))
            stack.enter_context(patch.object(workflow.RequestMeter, 'attach'))
            selected = stack.enter_context(patch.object(workflow, 'connect_target', return_value=(Mock(), nullcontext(Mock()))))
            stack.enter_context(patch.object(workflow, 'active_snapshot'))
            store = stack.enter_context(patch.object(workflow, 'open_store')).return_value
            with open_retriever(ROOT, 'pinecone'):
                store.audit.assert_called_once_with()
            self.assertEqual(selected.call_args.kwargs, {'read_only': True})

    def test_controlled_reads_and_sdk_writes_select_different_transports(self):
        from types import SimpleNamespace
        from test_pinecone_cloud import cloud_settings, description
        from northstar_atlas.vector_store.pinecone_workflow import connect_target, RequestMeter
        client = SimpleNamespace(indexes=SimpleNamespace(describe=lambda **_: description()), index=unittest.mock.Mock())
        meter = RequestMeter()
        with connect_target(client, meter, cloud_settings(), read_only=True)[1] as index:
            self.assertIsInstance(index, DocumentsIndex)
        client.index.assert_not_called()
        with patch.object(meter, 'attach'):
            connect_target(client, meter, cloud_settings(), read_only=False)
        client.index.assert_called_once_with(host=description()['host'])

    def test_control_plane_errors_have_one_physical_attempt(self):
        from test_pinecone_cloud import cloud_settings
        from northstar_atlas.vector_store.pinecone_cloud import sdk_client, safe_call, CloudError
        for failure in (httpx.ConnectError('synthetic'), 429, 503):
            calls = []
            def send(_, request):
                calls.append(request)
                if isinstance(failure, Exception):
                    raise failure
                return httpx.Response(failure, json={'error': 'synthetic'}, request=request)
            with self.subTest(failure=type(failure).__name__ if isinstance(failure, Exception) else failure):
                with sdk_client(cloud_settings()) as client, patch.object(httpx.HTTPTransport, 'handle_request', send):
                    with self.assertRaises(CloudError):
                        safe_call('describe', client.indexes.describe, name=cloud_settings().name)
                self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
