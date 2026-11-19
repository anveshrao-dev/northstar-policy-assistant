"""Real isolated Chroma/pinned MiniLM with synthetic development budget and HTTP."""
from datetime import datetime, timezone
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_local_rag_integration as integration
from northstar_atlas import local_development as dev
from northstar_atlas.budget_transition import _encode, _sha
from northstar_atlas.development_pricing import SOURCE, month_expiry
from northstar_atlas.ui.runtime import Submission


@unittest.skipUnless(os.environ.get('NORTHSTAR_RUN_MODEL_TESTS')=='1','requires explicitly enabled cached MiniLM checks')
class DevelopmentRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.LocalRAGIntegrationTests.setUpClass()

    def setUp(self):
        f=integration.LocalRAGIntegrationTests('runTest'); f.setUp(); self.addCleanup(f.doCleanups)
        self.f=f
        # Existing fixture prepares neither a one-shot nor a development authority.
        from northstar_atlas import local_test_authority as local
        for key in (local.ENV_PLAN,local.ENV_RUN,'NORTHSTAR_BUDGET_SCOPE','NORTHSTAR_BILLING_SCOPE',
                    'NORTHSTAR_BUDGET_MAX_ATTEMPTS','NORTHSTAR_BUDGET_MAX_RESERVED_USD'):
            os.environ.pop(key,None)
        now=datetime.now(timezone.utc)
        price=dict(id='synthetic-month',input_per_million_usd='0.10',output_per_million_usd='0.30',
                   source=SOURCE,confirmed_at=now.isoformat(),valid_until=month_expiry(now).isoformat())
        preview=dev.prepare(f.root,price,verified_pricing_sha256=_sha(_encode(price)),scope_review_sha256='a'*64,
                            approve_preparation=True,writers_stopped=True)
        dev.apply(dev.open_review(f.root),approved_plan_sha256=preview.plan_sha256,
                  approved_candidate_sha256=preview.candidate_sha256,approve_application=True,writers_stopped=True)
        os.environ.update({dev.ENV:'1',dev.PLAN_ENV:preview.plan_sha256,dev.RUN_ENV:dev.run_approval(preview.plan_sha256)})
        f.budget=dev.configured_development(f.root); f.f.budget=f.budget
        # Integration transport asserts the newest workflow, not the fixture's first one.
        def reserved(dispatched):
            import json
            flow=json.loads(f.budget.path.read_bytes())['workflows'][-1]
            self.assertEqual(flow['status'],'reserved')
            self.assertEqual(len(flow['calls']),int(dispatched))
        f.assert_reserved=reserved

    def submit(self,ident,question,mode='approved_only'):
        return self.f.owner().submit(Submission(ident,question,mode,'chroma',self.f.f.spec.reference_date),consent=True)

    def test_three_independent_questions_actual_retrieval_and_settled_citations(self):
        import json
        f=self.f
        from northstar_atlas.ui import runtime
        project=runtime.project_result
        def recorded(result,mode):
            flow=json.loads(f.budget.path.read_bytes())['workflows'][-1]
            self.assertEqual(flow['status'],'closed')
            saved=json.loads((f.root/f'.local/linear-demo-results/{flow["id"]}.json').read_bytes())
            self.assertEqual(saved['status'],'accepted')
            return project(result,mode)
        with patch.object(runtime,'project_result',side_effect=recorded):
            for i,question in enumerate(('How should an employee request time off and who approves it?',
                'For a planned internal transfer, how far in advance must HR notify IT Operations?',
                'What approvals are required for platform change management?')):
                view=self.submit(f'dev-{i}',question)
                self.assertEqual(view.kind,'answer',view)
                self.assertTrue(view.citations); self.assertTrue(view.sources)
        self.assertEqual(f.sent,['POST']*3)
        self.assertEqual(f.budget.inspect_history().attempts_consumed,3)
        self.assertEqual(str(f.budget.inspect_history().reserved_usd),'0.011061')
        self.assertEqual(self.submit('dev-0','Different replay question?').code,'duplicate')
        self.assertEqual(len(f.sent),3)

    def test_drafts_abstention_invalid_citation_and_envelope(self):
        f=self.f
        view=self.submit('draft','An engineer needs temporary production access to troubleshoot an AKS issue. Who should approve it, and which team grants the access?','draft_review')
        self.assertEqual(view.kind,'answer'); self.assertTrue(view.proposed)
        for ident,behavior,kind,code in (('unknown','abstain','abstain',None),
                ('bad-citation','citation','error','response_rejected'),
                ('bad-envelope','envelope','error','provider_failed'),
                ('timeout','timeout','error','provider_failed')):
            f.behavior=behavior
            view=self.submit(ident,'What is the corporate policy for lunar tourism?' if behavior=='abstain'
                             else 'How should an employee request time off and who approves it?')
            self.assertEqual((view.kind,view.code),(kind,code))
        self.assertEqual(len(f.sent),5)
        self.assertEqual(f.budget.inspect_history().attempts_consumed,5)

    def test_settlement_failure_withholds_answer_and_blocks_restart(self):
        f=self.f
        with patch.object(f.budget,'_write',wraps=f.budget._write) as write:
            original=f.budget._settle
            def failed(value,**kwargs):
                with patch.object(f.budget,'_write',side_effect=OSError('synthetic settlement failure')):
                    return original(value,**kwargs)
            with patch.object(f.budget,'_settle',side_effect=failed):
                self.assertEqual(self.submit('fails','How should an employee request time off and who approves it?').code,'budget_unavailable')
        self.assertTrue(f.budget.pending_path.exists())
        self.assertEqual(self.submit('another','What is the transfer policy?').code,'budget_unavailable')
        self.assertEqual(f.sent,['POST'])
