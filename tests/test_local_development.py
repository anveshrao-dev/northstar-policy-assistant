"""Monthly development accounting, using disposable synthetic checkouts only."""
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
import importlib
import os
import unittest
from unittest.mock import patch

from checkout_fixture import checkout
from budget_fixture import Provider, request
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, WorkflowRequest, LINEAR, _pricing


class CalendarTests(unittest.TestCase):
    def test_month_end_and_leap_years_use_calendar_arithmetic(self):
        module = importlib.import_module('northstar_atlas.development_pricing')
        for start, expected in (
            ('2027-01-31T12:34:56+00:00', '2027-02-28T12:34:56+00:00'),
            ('2028-01-31T12:34:56+00:00', '2028-02-29T12:34:56+00:00'),
            ('2028-02-29T12:34:56+00:00', '2028-03-29T12:34:56+00:00'),
            ('2026-03-31T12:34:56+00:00', '2026-04-30T12:34:56+00:00'),
            ('2026-12-31T12:34:56+00:00', '2027-01-31T12:34:56+00:00'),
            ('2026-01-31T23:34:56-05:00', '2026-03-01T04:34:56+00:00')):
            with self.subTest(start=start):
                self.assertEqual(module.month_expiry(datetime.fromisoformat(start)).isoformat(), expected)
        with self.assertRaises(ValueError): module.month_expiry(datetime(2026, 1, 1))

    def test_existing_policy_cannot_raise_cost_limit(self):
        with self.assertRaises(ValueError): BudgetPolicy(100, Decimal('2.00'))


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.dev = importlib.import_module('northstar_atlas.local_development')
        self.root = checkout(self)
        (self.root/'.local').mkdir(mode=0o700)
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        os.environ['NORTHSTAR_PROJECT_ROOT'] = str(self.root)
        self.now = datetime(2028, 1, 31, 12, 0, tzinfo=timezone.utc)
        self.price = dict(id='synthetic-price-1', input_per_million_usd='0.10', output_per_million_usd='0.30',
            source='https://nebius.com/services/token-factory/models/qwen-models-inference',
            confirmed_at=self.now.isoformat(), valid_until='2028-02-29T12:00:00+00:00')

    def prepare(self):
        from northstar_atlas.budget_transition import _encode, _sha
        self.preview = self.dev.prepare(self.root, self.price, verified_pricing_sha256=_sha(_encode(self.price)),
            scope_review_sha256='a'*64, approve_preparation=True, writers_stopped=True, clock=lambda:self.now)
        return self.preview

    def activate(self):
        self.prepare()
        b = self.dev.open_review(self.root, clock=lambda:self.now)
        self.dev.apply(b, approved_plan_sha256=self.preview.plan_sha256,
            approved_candidate_sha256=self.preview.candidate_sha256, approve_application=True, writers_stopped=True)
        self.b = self.dev.open_review(self.root, approval=self.preview.plan_sha256,
            run_approval=self.dev.run_approval(self.preview.plan_sha256), clock=lambda:self.now)
        return self.b

    def test_preparation_is_inert_and_cannot_collide_with_one_shot(self):
        self.prepare()
        b=self.dev.open_review(self.root, clock=lambda:self.now)
        with self.assertRaises(ValueError): b.check_ready(profile=LINEAR)
        self.assertFalse((self.root/'.local/controlled-live').exists())
        self.assertTrue(b.path.is_relative_to(self.root/'.local/development-authority'))
        with self.assertRaises((ValueError, FileExistsError)): self.prepare()

    def test_monthly_price_expires_and_legacy_validator_is_unchanged(self):
        b=self.activate()
        with b._locked():
            self.assertEqual(b._read()[2].reserved_usd, 0)
        price={k:v for k,v in self.price.items() if k!='id'}
        with self.assertRaises(ValueError): _pricing(price, self.now)
        self.now=datetime(2028,2,29,12,0,tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            with b._locked(): b._read()

    def test_wrong_digest_and_duplicate_application_denied(self):
        self.prepare(); b=self.dev.open_review(self.root, clock=lambda:self.now)
        with self.assertRaises(ValueError):
            self.dev.apply(b, approved_plan_sha256='b'*64, approved_candidate_sha256=self.preview.candidate_sha256,
                           approve_application=True, writers_stopped=True)
        self.dev.apply(b, approved_plan_sha256=self.preview.plan_sha256, approved_candidate_sha256=self.preview.candidate_sha256,
                       approve_application=True, writers_stopped=True)
        with self.assertRaises(ValueError):
            self.dev.apply(b, approved_plan_sha256=self.preview.plan_sha256, approved_candidate_sha256=self.preview.candidate_sha256,
                           approve_application=True, writers_stopped=True)

    def authorize_local(self):
        from types import SimpleNamespace
        from northstar_atlas import live_access, local_live
        os.environ.update(NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_UI_ENABLE_LIVE='1',
            NORTHSTAR_UI_ENABLE_GRAPH='0', NORTHSTAR_UI_ENABLE_PINECONE='0', VECTOR_STORE='chroma')
        self.enterContext(patch('platform.system', return_value='Darwin'))
        self.enterContext(live_access.local_session(self.root,backend='chroma',port=8501,enable_graph=False,approved=True))
        opts=local_live.server_options(8501)
        self.enterContext(patch('streamlit.config.get_option', side_effect=lambda k:opts.get(k.replace('.','_'))))
        self.enterContext(patch('streamlit.runtime.context._get_client_context', return_value=SimpleNamespace(
            remote_ip='127.0.0.1',headers=[('Host','127.0.0.1:8501'),('Origin','http://127.0.0.1:8501')])))

    def spec(self, ident='question-1', **changes):
        from dataclasses import replace
        return replace(WorkflowRequest(ident,LINEAR,'What is the transfer policy?','approved_only','chroma',date(2026,11,11)),**changes)

    def test_three_requests_duplicate_restart_and_conservative_no_generation_count(self):
        b=self.activate(); self.authorize_local()
        for i in range(3):
            with b.reserve(self.spec(str(i))): pass
        state=b.inspect_history()
        self.assertEqual(state.attempts_consumed,3)
        self.assertEqual(state.reserved_usd,Decimal('0.011061'))
        restarted=self.dev.open_review(self.root,approval=b.approval,run_approval=b.run_approval,clock=lambda:self.now)
        with self.assertRaises(ValueError):
            with restarted.reserve(self.spec('0')): self.fail('duplicate executed')
        self.assertEqual(restarted.inspect_history(),state)

    def test_hundred_questions_exhaust_even_without_generation(self):
        b=self.activate(); self.authorize_local()
        for i in range(100):
            with b.reserve(self.spec(str(i))): pass
        self.assertEqual(b.capacity().remaining_questions,0)
        self.assertEqual(b.capacity().reserved_usd,Decimal('0.368700'))
        with self.assertRaises(ValueError):
            with b.reserve(self.spec('101')): self.fail('limit bypass')

    def test_cost_limit_blocks_before_one_hundred_questions(self):
        self.price.update(input_per_million_usd='30',output_per_million_usd='30')
        b=self.activate(); self.authorize_local()
        with b.reserve(self.spec('1')): pass
        with b.reserve(self.spec('2')): pass
        self.assertFalse(b.capacity().available)
        self.assertEqual(b.capacity().reserved_usd,Decimal('1.966080'))
        with self.assertRaises(ValueError):
            with b.reserve(self.spec('3')): self.fail('cost bypass')

    def test_renewal_preserves_history_and_rejects_stale_duplicate_approval(self):
        from northstar_atlas.budget_transition import _encode,_sha
        b=self.activate(); self.authorize_local()
        with b.reserve(self.spec()) as r:
            with self.assertRaises(TimeoutError): r.invoke(Provider(TimeoutError()),'generator',0,request())
        before=b.inspect_history()
        source=b.path.read_bytes()
        old_files={p:p.read_bytes() for p in b.directory.rglob('*') if p.is_file() and p!=b.path}
        self.now=datetime(2028,3,1,12,0,tzinfo=timezone.utc)
        with self.assertRaises(ValueError): b.capacity()
        price=dict(self.price,id='price-2',confirmed_at=self.now.isoformat(),valid_until='2028-04-01T12:00:00+00:00')
        preview=self.dev.prepare_renewal(b,price,previous_plan_sha256=b.approval,
            verified_pricing_sha256=_sha(_encode(price)),approve_preparation=True,writers_stopped=True)
        self.assertEqual(b.path.read_bytes(),source)
        with self.assertRaises(ValueError): b.capacity()
        self.dev.apply(b,approved_plan_sha256=preview.plan_sha256,approved_candidate_sha256=preview.candidate_sha256,
                       approve_application=True,writers_stopped=True)
        with self.assertRaises(ValueError): b.capacity()
        fresh=self.dev.open_review(self.root,approval=preview.plan_sha256,
            run_approval=self.dev.run_approval(preview.plan_sha256),clock=lambda:self.now)
        self.assertEqual(fresh.inspect_history(),before)
        for p,raw in old_files.items(): self.assertEqual(p.read_bytes(),raw)
        with self.assertRaises(ValueError):
            self.dev.apply(fresh,approved_plan_sha256=preview.plan_sha256,approved_candidate_sha256=preview.candidate_sha256,
                           approve_application=True,writers_stopped=True)
        with fresh.reserve(self.spec('next')): pass
        self.assertEqual(fresh.inspect_history().attempts_consumed,2)

    def test_no_authorization_and_wrong_backend_fail_before_reservation(self):
        b=self.activate(); raw=b.path.read_bytes()
        with self.assertRaises(ValueError):
            with b.reserve(self.spec()): self.fail('unauthorized')
        self.authorize_local()
        for changes in ({'backend':'pinecone'}, {'profile':'llm_routing_grading_semantic_v1'}):
            with self.assertRaises(ValueError):
                with b.reserve(self.spec(**changes)): self.fail('wrong path')
        self.assertEqual(raw,b.path.read_bytes())

    def test_lock_contention_blocks_other_process(self):
        import subprocess
        b=self.activate()
        with b._physical_lock():
            code='import fcntl,os,sys; f=open(sys.argv[1],"r"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
            result=subprocess.run([os.sys.executable,'-I','-B','-c',code,str(b.lock_path)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
        self.assertEqual(b.inspect_history().attempts_consumed,0)

    def test_failed_activation_retains_guard_and_cannot_retry(self):
        self.prepare(); b=self.dev.open_review(self.root,clock=lambda:self.now)
        with patch.object(b,'_write',side_effect=OSError('synthetic write failure')),self.assertRaises(OSError):
            self.dev.apply(b,approved_plan_sha256=self.preview.plan_sha256,
                approved_candidate_sha256=self.preview.candidate_sha256,approve_application=True,writers_stopped=True)
        self.assertTrue(b.pending_path.exists())
        fresh=self.dev.open_review(self.root,clock=lambda:self.now)
        with self.assertRaises(ValueError):
            self.dev.apply(fresh,approved_plan_sha256=self.preview.plan_sha256,
                approved_candidate_sha256=self.preview.candidate_sha256,approve_application=True,writers_stopped=True)

    def test_tampered_candidate_rejected_without_ledger_write(self):
        import json
        self.prepare(); b=self.dev.open_review(self.root,clock=lambda:self.now); raw=b.path.read_bytes()
        target=b.directory/'review-0000/candidate.json'
        v=json.loads(target.read_bytes()); v['max_attempts']=101; target.write_text(json.dumps(v))
        with self.assertRaises(ValueError):
            self.dev.apply(b,approved_plan_sha256=self.preview.plan_sha256,
                approved_candidate_sha256=self.preview.candidate_sha256,approve_application=True,writers_stopped=True)
        self.assertEqual(raw,b.path.read_bytes())


class DevelopmentUITests(unittest.TestCase):
    def setUp(self):
        import test_local_live
        from northstar_atlas import local_development as dev
        self.dev=dev
        f=test_local_live.LocalLiveTests('runTest'); f.setUp(); self.addCleanup(f.doCleanups)
        self.f=f; self.historical=f.path.read_bytes()
        now=datetime.now(timezone.utc)
        from northstar_atlas.development_pricing import month_expiry,SOURCE
        from northstar_atlas.budget_transition import _encode,_sha
        price=dict(id='ui-price',input_per_million_usd='0.10',output_per_million_usd='0.30',source=SOURCE,
                   confirmed_at=now.isoformat(),valid_until=month_expiry(now).isoformat())
        preview=dev.prepare(f.root,price,verified_pricing_sha256=_sha(_encode(price)),scope_review_sha256='a'*64,
                            approve_preparation=True,writers_stopped=True)
        b=dev.open_review(f.root)
        dev.apply(b,approved_plan_sha256=preview.plan_sha256,approved_candidate_sha256=preview.candidate_sha256,
                  approve_application=True,writers_stopped=True)
        for key in ('NORTHSTAR_BUDGET_SCOPE','NORTHSTAR_BILLING_SCOPE','NORTHSTAR_BUDGET_MAX_ATTEMPTS'):
            os.environ.pop(key,None)
        os.environ.update({dev.ENV:'1',dev.PLAN_ENV:preview.plan_sha256,dev.RUN_ENV:dev.run_approval(preview.plan_sha256)})
        from northstar_atlas.live_budget import configured_budget
        f.budget=configured_budget(f.root); f.path=f.budget.path

    def test_three_independent_ui_questions_and_restart_duplicate_record_before_display(self):
        from dataclasses import replace
        f=self.f; owner=f.owner()
        for i in range(3):
            view=owner.submit(replace(f.submission(str(i)),question=f'Independent question {i}?'),consent=True)
            self.assertEqual(view.kind,'answer',view)
        self.assertEqual(f.budget.inspect_history().attempts_consumed,3)
        self.assertEqual(f.owner().submit(f.submission('0'),consent=True).code,'duplicate')
        self.assertEqual(len(f.calls),3)
        self.assertEqual((f.root/'.local/controlled-live/budget.json').read_bytes(),self.historical)

    def test_more_than_five_questions_not_limited_by_process_counter(self):
        owner=self.f.owner()
        for i in range(6): self.assertEqual(owner.submit(self.f.submission(str(i)),consent=True).kind,'answer')
        self.assertEqual(len(self.f.calls),6)

    def test_development_ui_status_and_consent(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from pathlib import Path
        import sys
        f=self.f; owner=f.owner()
        original=sys.modules['__main__']; self.addCleanup(sys.modules.__setitem__,'__main__',original)
        st.cache_resource.clear(); self.addCleanup(st.cache_resource.clear)
        with patch.object(Path,'cwd',return_value=f.root),patch('northstar_atlas.ui.runtime.ResourceOwner',return_value=owner):
            app=AppTest.from_file(str(f.root/'src/northstar_atlas/ui/app.py')).run(timeout=30)
            app.radio(key='experience').set_value('Live Q&A').run()
            self.assertTrue(app.chat_input[0].disabled)
            self.assertTrue(any('100' in v.value and 'remaining' in v.value.lower() for v in app.caption))
            app.checkbox(key='paid_opt_in').check().run()
            app.chat_input[0].set_value('A synthetic question?').run()
            self.assertEqual(app.session_state['conversation'].history[-1].view.kind,'answer')
            app.run(); self.assertEqual(len(f.calls),1)

    def test_revoked_development_approval_cannot_reach_transport(self):
        f=self.f
        import httpx
        from northstar_atlas.nebius import NebiusProvider, NebiusSettings, ProviderError
        from budget_fixture import request
        from northstar_atlas.live_budget import WorkflowRequest, LINEAR
        sent=[]
        def transport(req):
            sent.append(req.method)
            return httpx.Response(500)
        with NebiusProvider(NebiusSettings('synthetic-development-key'),allow_paid=True,
                            transport=httpx.MockTransport(transport)) as provider:
            original=provider.generate_budgeted
            def revoked(req,permit):
                with patch.dict(os.environ,{self.dev.RUN_ENV:'invalid'}):
                    return original(req,permit)
            with f.budget.reserve(WorkflowRequest('revoke',LINEAR,'Question?','approved_only','chroma',date.today())) as permit:
                with patch.object(provider,'generate_budgeted',side_effect=revoked),self.assertRaises(ProviderError):
                    permit.invoke(provider,'generator',0,request())
        self.assertEqual(sent,[])
        self.assertEqual(f.budget.inspect_history().attempts_consumed,1)

    def test_mixed_one_shot_configuration_cannot_fall_back(self):
        f=self.f
        with patch.dict(os.environ,NORTHSTAR_LOCAL_TEST_PLAN='a'*64):
            self.assertEqual(f.owner().submit(f.submission('mixed'),consent=True).code,'budget_unavailable')
        self.assertFalse(f.calls)

    def test_no_consent_does_not_create_result_or_call(self):
        f=self.f
        self.assertEqual(f.owner().submit(f.submission('no-consent'),consent=False).code,'opt_in_required')
        self.assertFalse(f.calls)
        self.assertEqual(f.budget.inspect_history().attempts_consumed,0)

    def test_development_launcher_requires_flag_and_chroma_without_graph(self):
        from northstar_atlas import local_live
        from streamlit.web import bootstrap
        from pathlib import Path
        with patch.object(Path,'cwd',return_value=self.f.root),patch.object(bootstrap,'run') as run:
            self.assertEqual(local_live.main(['--approve-local-live']),1)
            self.assertEqual(local_live.main(['--approve-local-live','--development','--enable-graph']),1)
            self.assertEqual(local_live.main(['--approve-local-live','--development','--backend','pinecone']),1)
            run.assert_not_called()

    def test_expiry_between_reservation_and_generation_denies_provider(self):
        f=self.f
        from northstar_atlas.live_budget import WorkflowRequest,LINEAR
        from budget_fixture import Provider,request
        import json
        price=json.loads(f.budget.path.read_bytes())['pricing_revisions'][-1]
        expiry=datetime.fromisoformat(price['valid_until'])
        provider=Provider()
        with f.budget.reserve(WorkflowRequest('expires',LINEAR,'Question?','approved_only','chroma',date.today())) as permit:
            with patch.object(f.budget,'clock',return_value=expiry),self.assertRaises(ValueError):
                permit.invoke(provider,'generator',0,request())
        self.assertFalse(provider.calls)
        self.assertEqual(f.budget.inspect_history().attempts_consumed,1)

    def test_concurrent_submissions_do_not_queue_or_double_charge(self):
        from contextlib import contextmanager
        from threading import Event,Thread
        from types import SimpleNamespace
        f=self.f; first=f.owner(); original=first._opener
        entered,release=Event(),Event(); results=[]
        @contextmanager
        def paused(root,**options):
            with original(root,**options) as service:
                def ask(*args,**kwargs):
                    entered.set()
                    if not release.wait(10): raise RuntimeError('synthetic synchronization timeout')
                    return service.ask(*args,**kwargs)
                yield SimpleNamespace(ask=ask)
        first._opener=paused
        worker=Thread(target=lambda:results.append(first.submit(f.submission('concurrent-1'),consent=True)))
        worker.start()
        try:
            self.assertTrue(entered.wait(10))
            self.assertEqual(f.owner().submit(f.submission('concurrent-2'),consent=True).code,'budget_unavailable')
        finally:
            release.set(); worker.join(15)
        self.assertFalse(worker.is_alive())
        self.assertEqual([result.kind for result in results],['answer'])
        self.assertEqual(len(f.calls),1)
        self.assertEqual(f.budget.inspect_history().attempts_consumed,1)
