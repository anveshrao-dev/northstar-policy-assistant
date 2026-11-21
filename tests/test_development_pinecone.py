"""Backend amendments use synthetic publications, real locks and no provider IO."""
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
import unittest
from unittest.mock import patch

import test_local_development as development_fixtures
from test_qa import FakeLLM, FixtureStore
from northstar_atlas.budget_transition import _encode, _sha
from northstar_atlas.live_budget import LINEAR


def publication(root):
    from northstar_atlas.embeddings import EmbeddedChunk, EmbeddingModel
    from northstar_atlas.ingestion.document_loader import load_documents
    from northstar_atlas.ingestion.chunker import chunk_document
    from northstar_atlas.local_embeddings import MODEL_ID, MODEL_REVISION
    from northstar_atlas.vector_store.records import IndexConfig
    from northstar_atlas.vector_store.preparation import freeze_snapshot
    from northstar_atlas.vector_store.pinecone_store import PineconeStore, PineconeTarget
    from northstar_atlas.vector_store import pinecone_workflow as w
    from northstar_atlas.vector_store.pinecone_cloud import record_confirmation
    from test_pinecone_cloud import cloud_settings
    model = EmbeddingModel(MODEL_ID, MODEL_REVISION, 384, 'l2')
    records = []
    for doc in load_documents(root, mode='draft_review'):
        chunk = chunk_document(doc).chunks[0]
        for i in range(w.EXPECTED_COUNTS[doc.document_id]):
            records.append(EmbeddedChunk(replace(chunk, chunk_id=f'{doc.document_id}-synthetic-{i}'),
                                         (1.,) + (0.,)*383, model))
    snapshot = freeze_snapshot(records, IndexConfig(model, 256), w.EXPECTED_CORPUS)
    w.save_snapshot(root, snapshot)
    settings = cloud_settings()
    record_confirmation(root, settings, project_id=settings.project_id)
    target = PineconeTarget(settings.name, 'https://synthetic.svc.pinecone.io', settings.project_id)
    store = PineconeStore(root, snapshot, target, documents=None, describe_target=None,
                         current_fingerprint=lambda:w.EXPECTED_CORPUS)
    metadata = {k:getattr(settings,k) for k in ('name','cloud','region','organization','project_name','project_id','plan')}
    (root/'.local/pinecone/target.json').write_text(json.dumps(metadata))
    with store.state.lock():
        store.state.write_run(snapshot.namespace, store._manifest('ready'))
        store.state.publish(snapshot.namespace)
    return store


class AmendmentTests(unittest.TestCase):
    setUp = development_fixtures.DevelopmentTests.setUp
    prepare = development_fixtures.DevelopmentTests.prepare
    activate = development_fixtures.DevelopmentTests.activate
    spec = development_fixtures.DevelopmentTests.spec

    def local(self, backend):
        from types import SimpleNamespace
        from northstar_atlas import live_access, local_live
        os.environ.update(NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_UI_ENABLE_LIVE='1',
            NORTHSTAR_UI_ENABLE_GRAPH='0', NORTHSTAR_UI_ENABLE_PINECONE=str(int(backend=='pinecone')), VECTOR_STORE=backend)
        self.enterContext(patch('platform.system', return_value='Darwin'))
        self.enterContext(live_access.local_session(self.root, backend=backend, port=8501, enable_graph=False, approved=True))
        opts=local_live.server_options(8501)
        from streamlit.config import get_option
        self.enterContext(patch('streamlit.config.get_option', side_effect=lambda k:
            opts[k.replace('.','_')] if k.replace('.','_') in opts else get_option(k)))
        self.enterContext(patch('streamlit.runtime.context._get_client_context', return_value=SimpleNamespace(
            remote_ip='127.0.0.1', headers=[('Host','127.0.0.1:8501'),('Origin','http://127.0.0.1:8501')])))

    def amendment(self):
        self.assertTrue(callable(getattr(self.dev, 'prepare_pinecone', None)), 'backend amendment API missing')
        self.store = publication(self.root)
        self.original = {p:p.read_bytes() for p in self.b.directory.rglob('*') if p.is_file()}
        self.next = self.dev.prepare_pinecone(self.b, publication_id=self.store.state.path.name,
            previous_plan_sha256=self.b.approval, scope_review_sha256='b'*64,
            approve_preparation=True, writers_stopped=True)
        return self.next

    def apply_amendment(self):
        self.dev.apply(self.b, approved_plan_sha256=self.next.plan_sha256,
            approved_candidate_sha256=self.next.candidate_sha256, approve_application=True, writers_stopped=True)
        self.new = self.dev.open_review(self.root, approval=self.next.plan_sha256,
            run_approval=self.dev.run_approval(self.next.plan_sha256), clock=lambda:self.now)
        return self.new

    def test_amendment_preserves_history_limits_and_original_approval(self):
        self.activate(); self.local('chroma')
        with self.b.reserve(self.spec('prior')): pass
        before=self.b.inspect_history(); self.amendment()
        self.assertEqual(self.next.candidate_sha256, _sha(self.b.path.read_bytes()))
        with self.assertRaises(ValueError): self.b.capacity()
        b=self.apply_amendment()
        for path,raw in self.original.items(): self.assertEqual(path.read_bytes(),raw)
        self.assertEqual(b.inspect_history(),before)
        self.assertEqual(b.capacity().remaining_questions,99)
        self.assertEqual(b.capacity().remaining_usd,Decimal('1.996313'))
        with self.assertRaises(ValueError): self.b.capacity()
        with b.reserve(self.spec('chroma-still-works')): pass
        self.assertEqual(b.capacity().remaining_questions,98)

    def test_pinecone_requires_amendment_before_reservation(self):
        b=self.activate(); self.local('pinecone'); raw=b.path.read_bytes()
        with self.assertRaises(ValueError):
            with b.reserve(self.spec(backend='pinecone')): self.fail('unauthorized')
        self.assertEqual(b.path.read_bytes(),raw)

    def test_approved_pinecone_uses_shared_count_and_survives_restart(self):
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        with b.reserve(self.spec(backend='pinecone')) as reservation:
            options=b.retrieval_options('pinecone')
            self.assertEqual(options['snapshot_id'],self.store.snapshot.snapshot_id)
            reservation.check_setup(b,profile=LINEAR,backend='pinecone')
        fresh=self.dev.open_review(self.root,approval=b.approval,run_approval=b.run_approval,clock=lambda:self.now)
        self.assertEqual(fresh.capacity().remaining_questions,99)
        self.assertEqual(fresh.capacity().reserved_usd,Decimal('0.003687'))
        with self.assertRaises(ValueError):
            with fresh.reserve(self.spec(backend='pinecone')): self.fail('duplicate')

    def test_wrong_publication_target_pointer_and_snapshot_fail_before_reservation(self):
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        paths=[self.root/'.local/pinecone/target.json', self.store.state.path/'active.json',
               self.store.state.path/(self.store.namespace+'.json'), self.root/'.local/pinecone/target-confirmation.json']
        raw=b.path.read_bytes()
        for path in paths:
            original=path.read_bytes(); path.write_bytes(original+b' ')
            try:
                with self.assertRaises(ValueError):
                    with b.reserve(self.spec(backend='pinecone')): self.fail('mismatch')
                self.assertEqual(raw,b.path.read_bytes())
            finally: path.write_bytes(original)

    def test_wrong_digest_and_replay_do_not_apply(self):
        self.activate(); self.amendment(); raw=self.b.path.read_bytes()
        with self.assertRaises(ValueError):
            self.dev.apply(self.b,approved_plan_sha256='0'*64,approved_candidate_sha256=self.next.candidate_sha256,
                           approve_application=True,writers_stopped=True)
        self.assertEqual(raw,self.b.path.read_bytes())
        self.apply_amendment()
        with self.assertRaises(ValueError): self.apply_amendment()
        with self.assertRaises(ValueError):
            self.dev.prepare_pinecone(self.new,publication_id=self.store.state.path.name,
                previous_plan_sha256=self.new.approval,scope_review_sha256='b'*64,
                approve_preparation=True,writers_stopped=True)

    def test_tampered_plan_is_rejected(self):
        self.activate(); self.amendment()
        path=self.b.directory/'review-0001/plan.json'; value=json.loads(path.read_bytes())
        value['pinecone']['namespace']='dev-'+'0'*64; path.write_bytes(_encode(value))
        with self.assertRaises(ValueError): self.apply_amendment()
        self.assertFalse((self.b.directory/'review-0001/applied.json').exists())

    def test_changed_history_blocks_application(self):
        self.activate(); self.amendment()
        value=json.loads(self.b.path.read_bytes()); value['billing_scope']='wrong'
        self.b.path.write_bytes(_encode(value))
        with self.assertRaises(ValueError): self.apply_amendment()

    def test_partial_receipt_failure_leaves_durable_guard(self):
        self.activate(); self.amendment(); original=self.dev._write_private
        def fail(b,p,raw):
            if p.name=='applied.json': raise OSError('synthetic storage failure')
            return original(b,p,raw)
        with patch.object(self.dev,'_write_private',side_effect=fail), self.assertRaises(OSError): self.apply_amendment()
        self.assertTrue(self.b.pending_path.exists())
        fresh=self.dev.open_review(self.root,approval=self.next.plan_sha256,
            run_approval=self.dev.run_approval(self.next.plan_sha256),clock=lambda:self.now)
        with self.assertRaises(ValueError): fresh.capacity()

    def test_lock_contention_denies_application(self):
        self.activate(); self.amendment()
        with self.b._physical_lock(),self.assertRaises(ValueError): self.apply_amendment()
        self.assertFalse((self.b.directory/'review-0001/applied.json').exists())

    def test_expired_price_blocks_amendment_application(self):
        self.activate(); self.amendment()
        self.now=datetime(2028,2,29,12,tzinfo=timezone.utc)
        with self.assertRaises(ValueError): self.apply_amendment()

    def test_pricing_renewal_inherits_capability_and_cumulative_usage(self):
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        with b.reserve(self.spec(backend='pinecone')): pass
        before=b.inspect_history()
        self.now=datetime(2028,3,1,12,tzinfo=timezone.utc)
        price=dict(self.price,id='new-price',confirmed_at=self.now.isoformat(),valid_until='2028-04-01T12:00:00+00:00')
        next_price=self.dev.prepare_renewal(b,price,previous_plan_sha256=b.approval,
            verified_pricing_sha256=_sha(_encode(price)),approve_preparation=True,writers_stopped=True)
        self.dev.apply(b,approved_plan_sha256=next_price.plan_sha256,approved_candidate_sha256=next_price.candidate_sha256,
                       approve_application=True,writers_stopped=True)
        fresh=self.dev.open_review(self.root,approval=next_price.plan_sha256,
            run_approval=self.dev.run_approval(next_price.plan_sha256),clock=lambda:self.now)
        self.assertEqual(fresh.inspect_history(),before)
        with fresh.reserve(self.spec('second',backend='pinecone')): pass
        self.assertEqual(fresh.capacity().remaining_questions,98)


    def test_wrong_authority_and_stale_parent_denied_before_packet_creation(self):
        self.activate(); store=publication(self.root)
        for parent in ('0'*64,self.b.approval):
            old=self.b.billing_scope
            if parent==self.b.approval: self.b.billing_scope='another-authority'
            try:
                with self.assertRaises(ValueError):
                    self.dev.prepare_pinecone(self.b,publication_id=store.state.path.name,
                        previous_plan_sha256=parent,scope_review_sha256='b'*64,
                        approve_preparation=True,writers_stopped=True)
                self.assertFalse((self.b.directory/'review-0001').exists())
            finally: self.b.billing_scope=old

    def test_partial_preparation_fences_old_runtime_without_ledger_change(self):
        self.activate(); store=publication(self.root); raw=self.b.path.read_bytes()
        original=self.dev._write_private
        def fail(b,path,data):
            if path.name=='candidate.json': raise OSError('synthetic')
            return original(b,path,data)
        with patch.object(self.dev,'_write_private',side_effect=fail),self.assertRaises(OSError):
            self.dev.prepare_pinecone(self.b,publication_id=store.state.path.name,
                previous_plan_sha256=self.b.approval,scope_review_sha256='b'*64,
                approve_preparation=True,writers_stopped=True)
        self.assertEqual(raw,self.b.path.read_bytes())
        with self.assertRaises(ValueError): self.b.capacity()

    def test_publication_lock_contention_cannot_reserve(self):
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        before=b.path.read_bytes()
        with self.store.state.lock(), self.assertRaises((ValueError,OSError)):
            with b.reserve(self.spec(backend='pinecone')): self.fail('contended')
        self.assertEqual(before,b.path.read_bytes())

    def test_pending_publication_transition_blocks_without_reservation(self):
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        before=b.path.read_bytes(); (self.store.state.path/'restoration.pending').write_text('synthetic')
        with self.assertRaises(ValueError):
            with b.reserve(self.spec(backend='pinecone')): self.fail('pending')
        self.assertEqual(before,b.path.read_bytes())

    def test_pinecone_history_cannot_be_injected_into_chroma_only_chain(self):
        self.activate(); self.local('chroma')
        with self.b.reserve(self.spec()): pass
        value=json.loads(self.b.path.read_bytes()); value['workflows'][0]['backend']='pinecone'
        self.b.path.write_bytes(_encode(value))
        with self.assertRaises(ValueError): self.b.capacity()

    def test_invalid_snapshot_override_stops_before_credentials_or_retrieval(self):
        from northstar_atlas import qa,nebius
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        with b.reserve(self.spec(backend='pinecone')) as permit:
            with patch.object(nebius,'load_settings',side_effect=AssertionError('credential IO')), \
                 patch.object(qa,'open_retriever',side_effect=AssertionError('retrieval IO')):
                with self.assertRaises(qa.QAConfigurationError):
                    with qa.open_qa_service(self.root,backend='pinecone',snapshot_id='0'*64,
                                           allow_paid=True,budget=b,reservation=permit): self.fail('opened')
        self.assertEqual(json.loads(b.path.read_bytes())['workflows'][0]['calls'],[])

    def test_changed_publication_after_reservation_blocks_resource_initialization(self):
        from northstar_atlas import qa,nebius
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        with b.reserve(self.spec(backend='pinecone')) as permit:
            path=self.store.state.path/'active.json'; path.write_bytes(path.read_bytes()+b' ')
            with patch.object(nebius,'load_settings',side_effect=AssertionError('credential IO')), \
                 patch.object(qa,'open_retriever',side_effect=AssertionError('retrieval IO')):
                with self.assertRaises(qa.QAConfigurationError):
                    with qa.open_qa_service(self.root,backend='pinecone',allow_paid=True,budget=b,reservation=permit):
                        self.fail('opened')
        self.assertEqual(json.loads(b.path.read_bytes())['workflows'][0]['calls'],[])

    def test_chroma_renewal_remains_available_if_optional_publication_fails(self):
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('chroma')
        self.store.state.write_run(self.store.namespace,self.store._manifest('incomplete',phase='failed audit'))
        self.now=datetime(2028,3,1,12,tzinfo=timezone.utc)
        price=dict(self.price,id='new-price',confirmed_at=self.now.isoformat(),valid_until='2028-04-01T12:00:00+00:00')
        preview=self.dev.prepare_renewal(b,price,previous_plan_sha256=b.approval,
            verified_pricing_sha256=_sha(_encode(price)),approve_preparation=True,writers_stopped=True)
        self.dev.apply(b,approved_plan_sha256=preview.plan_sha256,approved_candidate_sha256=preview.candidate_sha256,
                       approve_application=True,writers_stopped=True)
        fresh=self.dev.open_review(self.root,approval=preview.plan_sha256,
            run_approval=self.dev.run_approval(preview.plan_sha256),clock=lambda:self.now)
        with fresh.reserve(self.spec('chroma')): pass
        self.assertEqual(fresh.capacity().remaining_questions,99)

    def test_expired_price_blocks_execution_and_keeps_full_reservation(self):
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        self.now=datetime(2028,2,29,12,tzinfo=timezone.utc); before=b.path.read_bytes()
        with self.assertRaises(ValueError):
            with b.reserve(self.spec(backend='pinecone')): self.fail('expired')
        self.assertEqual(before,b.path.read_bytes())

    def test_ui_requires_consent_and_records_one_settled_synthetic_answer(self):
        from types import SimpleNamespace
        from northstar_atlas.ui.runtime import ResourceOwner,Submission
        from northstar_atlas import qa
        from test_context import hit,chunk,AS_OF
        from test_qa import authored_claims
        from northstar_atlas.generation import prepare_answer
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        os.environ.update({self.dev.ENV:'1',self.dev.PLAN_ENV:b.approval,self.dev.RUN_ENV:b.run_approval})
        hits=[hit(chunk(doc='HR-001',status='Approved (Synthetic)',effective=AS_OF))]
        prepared=prepare_answer('What is the transfer policy?',hits,reference_date=AS_OF)
        provider=FakeLLM(authored_claims(prepared)); provider.requires_durable_budget=True
        seen=[]
        @contextmanager
        def retrieval(root,backend,**options):
            seen.append(options)
            self.assertEqual(options['snapshot_id'],self.store.snapshot.snapshot_id)
            self.assertEqual(options['pinecone_binding']['publication_id'],self.store.state.path.name)
            yield FixtureStore(hits),object(),'synthetic'
        @contextmanager
        def opener(root,**options):
            flow=json.loads(b.path.read_bytes())['workflows'][-1]
            self.assertEqual((flow['status'],flow['calls']),('reserved',[]))
            with qa.open_qa_service(root,provider=provider,**options) as service: yield service
        # Configured authority uses the fixture's clock, never the actual date.
        with patch.object(self.dev,'open_review',return_value=b),patch.object(qa,'open_retriever',retrieval):
            owner=ResourceOwner(self.root,live_enabled=True,pinecone_enabled=True,budget=b,opener=opener)
            self.addCleanup(owner.close)
            submission=Submission('ui-once','What is the transfer policy?','approved_only','pinecone',AS_OF)
            self.assertEqual(owner.submit(submission,consent=False).code,'opt_in_required')
            self.assertEqual(json.loads(b.path.read_bytes())['workflows'],[])
            view=owner.submit(submission,consent=True)
            self.assertEqual(view.kind,'answer',view)
            record=json.loads((self.root/'.local/linear-demo-results/ui-once.json').read_bytes())
            self.assertEqual(record['status'],'accepted')
            self.assertEqual(json.loads(b.path.read_bytes())['workflows'][0]['status'],'closed')
            self.assertFalse(b.pending_path.exists())
            self.assertEqual(owner.submit(submission,consent=True).code,'duplicate')
            self.assertEqual(len(provider.calls),1)
            self.assertEqual(len(seen),1)
            self.assertEqual(b.capacity().remaining_questions,99)


    def test_concurrent_application_has_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        self.activate(); self.amendment()
        def apply_once():
            try:
                self.apply_amendment()
                return 'applied'
            except ValueError:
                return 'denied'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:apply_once(),range(2)))
        self.assertEqual(sorted(results),['applied','denied'])
        self.assertFalse(self.b.pending_path.exists())
        self.assertEqual(json.loads(self.b.path.read_bytes())['workflows'],[])

    def test_streamlit_pinecone_selection_is_fixed_and_consent_starts_off(self):
        import sys
        from pathlib import Path
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from northstar_atlas.ui.runtime import ResourceOwner
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        os.environ.update({self.dev.ENV:'1',self.dev.PLAN_ENV:b.approval,self.dev.RUN_ENV:b.run_approval})
        owner=ResourceOwner(self.root,live_enabled=True,pinecone_enabled=True,budget=b,
                            opener=lambda *a,**k:self.fail('UI startup accessed providers'))
        self.addCleanup(owner.close)
        original=sys.modules['__main__']; self.addCleanup(sys.modules.__setitem__,'__main__',original)
        st.cache_resource.clear(); self.addCleanup(st.cache_resource.clear)
        with patch.object(Path,'cwd',return_value=self.root),patch.object(self.dev,'open_review',return_value=b), \
             patch('northstar_atlas.ui.runtime.ResourceOwner',return_value=owner):
            app=AppTest.from_file(str(self.root/'src/northstar_atlas/ui/app.py')).run(timeout=30)
            app.radio(key='experience').set_value('Live Q&A').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Standard RAG · Pinecone · monthly' in c.value for c in app.caption))
            self.assertEqual(app.selectbox(key='backend').options,['Pinecone · hosted'])
            self.assertTrue(app.selectbox(key='backend').disabled)
            self.assertFalse(app.checkbox(key='paid_opt_in').value)
            self.assertTrue(app.chat_input[0].disabled)
            app.checkbox(key='paid_opt_in').check().run()
            self.assertFalse(app.chat_input[0].disabled)
        self.assertEqual(json.loads(b.path.read_bytes())['workflows'],[])

    def test_actual_retriever_rejects_wrong_snapshot_before_audit_or_generation(self):
        from types import SimpleNamespace
        from northstar_atlas import qa
        from northstar_atlas.vector_store import pinecone_cloud as cloud, pinecone_store
        from test_pinecone_cloud import cloud_settings,description
        self.activate(); self.store=publication(self.root)
        value=description(); value['host']=self.store.target.host
        described=[]
        def describe(**kw): described.append('describe'); return value
        @contextmanager
        def client(settings):
            yield SimpleNamespace(_http=object(),indexes=SimpleNamespace(describe=describe))
        provider=FakeLLM()
        with patch('northstar_atlas.local_embeddings.MiniLMProvider',return_value=SimpleNamespace()), \
             patch.object(cloud,'load_settings',return_value=cloud_settings()),patch.object(cloud,'sdk_client',client), \
             patch.object(pinecone_store.PineconeStore,'audit',side_effect=AssertionError('audit before snapshot validation')):
            with self.assertRaises(qa.QAConfigurationError):
                with qa.open_qa_service(self.root,backend='pinecone',snapshot_id='0'*64,provider=provider):
                    self.fail('wrong snapshot opened')
        self.assertEqual(provider.calls,[])
        self.assertEqual(described,['describe'])

    def test_bound_target_mismatch_stops_before_description(self):
        from northstar_atlas import qa
        from northstar_atlas.vector_store import pinecone_cloud as cloud
        from test_pinecone_cloud import cloud_settings
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        with b.reserve(self.spec(backend='pinecone')):
            options=b.retrieval_options('pinecone')
            settings=replace(cloud_settings(),name='another-index')
            with patch('northstar_atlas.local_embeddings.MiniLMProvider',return_value=object()), \
                 patch.object(cloud,'load_settings',return_value=settings),patch.object(cloud,'check_confirmation'), \
                 patch.object(cloud,'sdk_client',side_effect=AssertionError('remote description')):
                with self.assertRaises(ValueError):
                    with qa.open_retriever(self.root,'pinecone',**options): self.fail('wrong target')

    def test_cli_prepares_only_with_exact_required_flags(self):
        from pathlib import Path
        from contextlib import redirect_stdout
        import io
        self.activate(); store=publication(self.root); original=self.dev.open_review
        with patch.object(Path,'cwd',return_value=self.root), \
             patch.object(self.dev,'open_review',side_effect=lambda root:original(root,clock=lambda:self.now)), \
             redirect_stdout(io.StringIO()) as output:
            result=self.dev.main(['prepare-pinecone','--publication-id',store.state.path.name,
                '--previous-plan-sha256',self.b.approval,'--scope-review-sha256','b'*64,
                '--writers-stopped','--approve-preparation'])
        self.assertEqual(result,0)
        packet=json.loads(output.getvalue())
        self.assertEqual(packet['candidate_sha256'],_sha(self.b.path.read_bytes()))
        self.assertFalse((self.b.directory/'review-0001/applied.json').exists())


    def test_post_setup_binding_change_blocks_final_transport_but_allows_settlement(self):
        import httpx
        from northstar_atlas.nebius import NebiusProvider,NebiusSettings,ProviderError
        from budget_fixture import request
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        sent=[]
        def send(req):
            sent.append(req.method)
            return httpx.Response(500)
        with NebiusProvider(NebiusSettings('synthetic-fixture-key'),allow_paid=True,
                            transport=httpx.MockTransport(send)) as provider:
            with b.reserve(self.spec(backend='pinecone')) as permit:
                b.retrieval_options('pinecone')
                path=self.root/'.local/pinecone/target-confirmation.json'
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaises(ProviderError): permit.invoke(provider,'generator',0,request())
        self.assertEqual(sent,[])
        flow=json.loads(b.path.read_bytes())['workflows'][0]
        self.assertEqual(flow['status'],'uncertain')
        self.assertFalse(b.pending_path.exists())
        self.assertEqual(len(flow['calls']),1)

    def test_post_opener_binding_change_blocks_search(self):
        from northstar_atlas import qa
        self.activate(); self.amendment(); b=self.apply_amendment(); self.local('pinecone')
        store=FixtureStore(); provider=FakeLLM(); provider.requires_durable_budget=True
        @contextmanager
        def retrieval(*args,**kwargs): yield store,object(),'synthetic'
        spec=self.spec(backend='pinecone')
        with b.reserve(spec) as permit,patch.object(qa,'open_retriever',retrieval):
            with qa.open_qa_service(self.root,backend='pinecone',provider=provider,budget=b,reservation=permit) as service:
                path=self.store.state.path/(self.store.namespace+'.json')
                raw=path.read_bytes(); temporary=path.with_suffix('.replacement'); temporary.write_bytes(raw)
                os.replace(temporary,path)
                result=service.ask(spec.question,mode=spec.mode,reference_date=spec.reference_date,reservation=permit)
        self.assertFalse(result.accepted)
        self.assertEqual(store.calls,[])
        self.assertEqual(provider.calls,[])
        self.assertFalse(b.pending_path.exists())


    def test_launcher_accepts_only_applied_pinecone_capability(self):
        from pathlib import Path
        from northstar_atlas import local_live
        from streamlit.web import bootstrap
        self.activate(); self.amendment(); b=self.apply_amendment()
        for relative in ('.local/tmp','.local/cache','.venv/bin'):
            (self.root/relative).mkdir(parents=True,exist_ok=True)
        os.environ.update({self.dev.PLAN_ENV:b.approval,self.dev.RUN_ENV:b.run_approval})
        with patch('platform.system',return_value='Darwin'),patch.object(Path,'cwd',return_value=self.root), \
             patch.object(self.dev,'open_review',return_value=b),patch.object(bootstrap,'run') as run:
            result=local_live.main(['--approve-local-live','--development','--backend','pinecone'])
        self.assertEqual(result,0)
        self.assertEqual(run.call_count,1)
        self.assertEqual(json.loads(b.path.read_bytes())['workflows'],[])


class SnapshotForwardingTests(unittest.TestCase):
    def test_dense_explicit_snapshot_reaches_enforcement_boundary(self):
        from northstar_atlas import qa
        from test_context import ROOT
        seen=[]
        @contextmanager
        def retrieval(root,backend,*,snapshot_id=None):
            seen.append(snapshot_id)
            if snapshot_id!='a'*64: raise ValueError('wrong snapshot')
            yield FixtureStore(),object(),'synthetic'
        with patch.object(qa,'open_retriever',retrieval):
            with qa.open_qa_service(ROOT,backend='pinecone',provider=FakeLLM(),snapshot_id='a'*64) as service:
                result=service.ask('Test?')
        self.assertTrue(result.accepted)
        self.assertEqual(seen,['a'*64])
