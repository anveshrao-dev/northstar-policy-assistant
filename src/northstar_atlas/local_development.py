"""Explicit local development accounting; never reuses historical authorities.

Operator-only preparation/renewal is separate from loopback execution. Immutable
review packets and receipts retain approvals; interrupted operations stay blocked.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import timezone
from decimal import Decimal
import os
from pathlib import Path
import stat
import uuid

from .budget_transition import (_encode, _decode, _sha, _digest, _read_private,
                                _write_private, _sync, _private_directory)
from .development_pricing import validate_price
from .live_budget import (PersistentBudget, BudgetPolicy, BudgetError, LINEAR, _SLOTS,
                          _require, _estimate, _now, _timestamp)
from .llm import MODEL_ID
from .project import validate_project_root, contained_path

ROOT = '.local/development-authority'
ENV = 'NORTHSTAR_DEVELOPMENT_ENABLED'
PLAN_ENV = 'NORTHSTAR_DEVELOPMENT_PLAN'
RUN_ENV = 'NORTHSTAR_DEVELOPMENT_RUN_APPROVAL'
KIND = 'local-development-monthly-v1'
PREPARING = 'preparation.pending'
ACTIVATION = 'development-activation.json'


@dataclass(frozen=True)
class DevelopmentPolicy(BudgetPolicy):
    max_attempts: int = 100
    max_reserved_usd: Decimal = Decimal('2.000000')

    def __post_init__(self):
        _require(type(self.max_attempts) is int and self.max_attempts == 100
                 and self.max_reserved_usd == Decimal('2.000000'))


@dataclass(frozen=True)
class Preview:
    plan_sha256: str
    candidate_sha256: str
    estimated_next_request_usd: str


@dataclass(frozen=True)
class Capacity:
    remaining_questions: int
    reserved_usd: Decimal
    remaining_usd: Decimal
    next_request_usd: Decimal
    available: bool


def selected():
    # Invalid values and leftover selection fields must fail closed, not fall back.
    return os.environ.get(ENV, '0') != '0' or PLAN_ENV in os.environ or RUN_ENV in os.environ


def run_approval(plan_sha256):
    _digest(plan_sha256)
    return _sha(('local-development-run:'+plan_sha256).encode())


def _identity(path):
    info = path.lstat()
    return [info.st_dev, info.st_ino]


def _bindings(b):
    from .local_embeddings import MODEL_ID as EMBEDDING, MODEL_REVISION
    return dict(kind=KIND, scope=b.scope, billing_scope=b.billing_scope,
        model=MODEL_ID, backend='chroma', profile=LINEAR, modes=['approved_only', 'draft_review'],
        max_attempts=100, max_reserved_usd='2.000000',
        corpus_sha256=_sha(contained_path(b.root, 'data/corpus_manifest.json').read_bytes()),
        embedding_model=EMBEDDING, embedding_revision=MODEL_REVISION,
        identities=[_identity(p) for p in (b.root, b.root/'.local', b.authority_root,
                    b.authority_root/'.local', b.directory, b.lock_path)])


class DevelopmentBudget(PersistentBudget):
    """Fixed policy/path, with the original lock, permits and settlement protocol."""
    def __init__(self, root, *, scope, billing_scope, approval=None, run_approval=None, clock=_now):
        root = validate_project_root(root)
        authority = contained_path(root, ROOT)
        super().__init__(root, scope=scope, billing_scope=billing_scope, policy=DevelopmentPolicy(),
                         clock=clock, authority_root=authority, authorized_authority_root=authority)
        from .live_access import require_local_access
        self.before_request = require_local_access
        self.approval, self.run_approval = approval, run_approval
        self._selection = None

    def _paths(self):
        # Recheck revocation at resource setup and the final transport claim,
        # including changes while the workflow's exclusive lock is held.
        if getattr(self, '_selection', None) is not None:
            _require(self._selection == _selection())
        super()._paths()

    def _validate_price(self, price, now=None):
        validate_price(price, now)

    def _accounted_slots(self, flow, selected):
        # Every durable question counts, even if retrieval fails before generation.
        return _SLOTS[LINEAR]

    def _validate_document(self, value, *, current=True):
        _require(isinstance(value, dict) and value.get('version') == 2 and value.get('legacy') is None
                 and 'diagnostic_request_sha256' not in value)
        result = super()._validate_document(value, current=current)
        _require(all(f['profile'] == LINEAR and f['backend'] == 'chroma' for f in value['workflows']))
        return result

    @contextmanager
    def _locked(self, *, diagnostic_approval=None):
        _require(diagnostic_approval is None)
        if self._selection is not None:
            _require(self._selection == _selection())
        with self._physical_lock():
            _require(not os.path.lexists(self.directory/PREPARING))
            try:
                plan, candidate, _, _ = _chain(self, current=True)
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                raise BudgetError('development approval unavailable; inference denied') from None
            _require(self.approval == _sha(_encode(plan)) and self.run_approval == run_approval(self.approval))
            current = self._read()[0]
            _extends(candidate, current)
            yield None

    @contextmanager
    def reserve(self, request, *, diagnostic_approval=None):
        from .live_access import require_live_budget, require_local_request
        require_live_budget(self.root, self)
        require_local_request(self.root, backend='chroma', graph=False)
        _require(request.profile == LINEAR and request.backend == 'chroma' and len(request.question) <= 2000)
        with super().reserve(request, diagnostic_approval=diagnostic_approval) as reservation:
            yield reservation

    def check_ready(self, *, profile=None):
        _require(profile in (None, LINEAR))
        return super().check_ready(profile=LINEAR)

    def capacity(self):
        with self._locked():
            value, prices, state = self._read()
            cost = _estimate(prices[value['active_pricing']], 30720, 2048)
            remaining = 100-state.attempts_consumed
            money = Decimal('2')-state.reserved_usd
            return Capacity(remaining, state.reserved_usd, money, cost, remaining > 0 and money >= cost)


def _extends(previous, current):
    _require({k:v for k,v in previous.items() if k != 'workflows'} ==
             {k:v for k,v in current.items() if k != 'workflows'})
    _require(current['workflows'][:len(previous['workflows'])] == previous['workflows'])


def _packets(b):
    entries = sorted(b.directory.glob('review-*'))
    _require(1 <= len(entries) <= 1000 and [p.name for p in entries] ==
             [f'review-{i:04d}' for i in range(len(entries))])
    for p in entries: _private_directory(b, p)
    return entries


def _receipt(plan):
    return _encode(dict(plan_sha256=_sha(_encode(plan)), candidate_sha256=plan['candidate_sha256']))


def _packet(b, directory, *, previous=None, current=False):
    plan = _decode(_read_private(b, directory/'plan.json'))
    candidate_raw = _read_private(b, directory/'candidate.json')
    candidate, prices, state = b._validate_document(_decode(candidate_raw), current=current)
    _require(candidate_raw == _encode(candidate))
    source_raw = _read_private(b, directory/'source.json')
    source = _decode(source_raw)
    number = int(directory.name[7:])
    binding = _bindings(b)
    _require(plan == dict(binding, revision=number, parent_plan_sha256=previous[0] if previous else None,
        source_sha256=_sha(source_raw), source_ledger_identity=plan.get('source_ledger_identity'),
        candidate_sha256=_sha(candidate_raw), pricing_sha256=_sha(_encode(candidate['pricing_revisions'][-1])),
        scope_review_sha256=plan.get('scope_review_sha256')))
    _digest(plan['scope_review_sha256'])
    ident = plan['source_ledger_identity']
    _require(isinstance(ident, list) and len(ident)==2 and all(type(v) is int and v >= 0 for v in ident))
    if previous is None:
        _require(number == 0 and source == candidate and candidate['workflows'] == []
                 and len(candidate['pricing_revisions']) == 1)
    else:
        _require(number > 0)
        b._validate_document(source, current=False)
        _extends(previous[1], source)
        _require(not any(f['status']=='reserved' for f in source['workflows']))
        expected = deepcopy(source)
        expected['pricing_revisions'].append(candidate['pricing_revisions'][-1])
        expected['active_pricing'] = candidate['pricing_revisions'][-1]['id']
        _require(candidate == expected and plan['scope_review_sha256'] == previous[2])
        old_price = next(p for p in source['pricing_revisions'] if p['id'] == source['active_pricing'])
        _require(_timestamp(candidate['pricing_revisions'][-1]['confirmed_at']) > _timestamp(old_price['confirmed_at']))
    _require(candidate['active_pricing'] == candidate['pricing_revisions'][-1]['id'])
    cost = _estimate(prices[candidate['active_pricing']], 30720, 2048)
    _require(state.attempts_consumed < 100 and state.reserved_usd + cost <= Decimal('2'))
    return plan, candidate, cost


def _chain(b, *, current, allow_pending=False):
    previous = None
    entries = _packets(b)
    for index, directory in enumerate(entries):
        last = index == len(entries)-1
        plan, candidate, cost = _packet(b, directory, previous=previous, current=current and last)
        if not (last and allow_pending):
            _require(_read_private(b, directory/'applied.json') == _receipt(plan))
        previous = (_sha(_encode(plan)), candidate, plan['scope_review_sha256'])
        if index == 0 and not (last and allow_pending):
            _require(_read_private(b, b.directory/ACTIVATION) == _receipt(plan))
    return plan, candidate, cost, entries[-1]


def open_review(root, *, approval=None, run_approval=None, clock=_now):
    """Read only the designated development contract; never probe another ledger."""
    root=validate_project_root(root)
    # Scope discovery is not authorization. Full private path/contract checks follow.
    p=contained_path(root, ROOT+'/.local/controlled-live/review-0000/plan.json')
    info=p.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_uid==os.geteuid()
             and stat.S_IMODE(info.st_mode)==0o600 and info.st_nlink==1 and info.st_size <= 65536)
    with os.fdopen(os.open(p, os.O_RDONLY|os.O_NOFOLLOW), 'rb') as stream:
        _require(_identity(p)==[os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino])
        plan=_decode(stream.read(65537))
    b=DevelopmentBudget(root, scope=plan['scope'], billing_scope=plan['billing_scope'],
                        approval=approval, run_approval=run_approval, clock=clock)
    b._paths()
    return b


def _write_packet(b, number, source, candidate, scope_review, parent):
    directory=b.directory/f'review-{number:04d}'
    _require(not os.path.lexists(directory))
    plan=dict(_bindings(b), revision=number, parent_plan_sha256=parent,
        source_sha256=_sha(source), source_ledger_identity=_identity(b.path),
        candidate_sha256=_sha(_encode(candidate)),
        pricing_sha256=_sha(_encode(candidate['pricing_revisions'][-1])), scope_review_sha256=scope_review)
    directory.mkdir(mode=0o700); _sync(b.directory)
    for name, raw in (('source.json',source), ('candidate.json',_encode(candidate)), ('plan.json',_encode(plan))):
        _write_private(b,directory/name,raw)
    return plan


def prepare(root, pricing, *, verified_pricing_sha256, scope_review_sha256,
            approve_preparation, writers_stopped, clock=_now):
    _require(approve_preparation is True and writers_stopped is True)
    _digest(scope_review_sha256); _digest(verified_pricing_sha256)
    _require(_sha(_encode(pricing))==verified_pricing_sha256)
    validate_price({k:v for k,v in pricing.items() if k!='id'}, clock())
    root=validate_project_root(root)
    authority=contained_path(root,ROOT)
    _require(not os.path.lexists(authority))
    # Never create or inspect the historical .local/controlled-live authority.
    for parent in (root,root/'.local'):
        info=parent.lstat()
        _require(stat.S_ISDIR(info.st_mode) and info.st_uid==os.geteuid()
                 and not stat.S_IMODE(info.st_mode)&0o7022)
    from .local_test_authority import _create
    scope=uuid.uuid4().hex
    authority.mkdir(mode=0o700); _sync(authority.parent)
    (authority/'.local').mkdir(mode=0o700); _sync(authority)
    directory=authority/'.local/controlled-live'
    directory.mkdir(mode=0o700); _sync(directory.parent)
    value=dict(version=2,scope=scope,billing_scope='local-dev-'+scope,provider='nebius',model=MODEL_ID,
        max_attempts=100,max_reserved_usd='2.000000',legacy=None,pricing_revisions=[pricing],
        active_pricing=pricing['id'],workflows=[])
    _create(directory/PREPARING,b'local-development-inert-v1')
    _create(directory/'budget.lock',b''); _create(directory/'budget.json',_encode(value))
    b=DevelopmentBudget(root,scope=scope,billing_scope=value['billing_scope'],clock=clock)
    with b._physical_lock():
        b._validate_document(value)
        plan=_write_packet(b,0,_encode(value),value,scope_review_sha256,None)
        _,_,cost,_=_chain(b,current=True,allow_pending=True)
    return Preview(_sha(_encode(plan)),plan['candidate_sha256'],str(cost))


def prepare_renewal(b, pricing, *, previous_plan_sha256, verified_pricing_sha256,
                    approve_preparation, writers_stopped):
    _require(approve_preparation is True and writers_stopped is True)
    _digest(previous_plan_sha256); _digest(verified_pricing_sha256)
    _require(_sha(_encode(pricing))==verified_pricing_sha256)
    with b._physical_lock():
        _require(not os.path.lexists(b.directory/PREPARING))
        previous, candidate, _, directory=_chain(b,current=False)
        _require(_sha(_encode(previous))==previous_plan_sha256)
        source=b._read_bytes(); value,_,state=b._validate_document(_decode(source),current=False)
        _extends(candidate,value)
        _require(not any(f['status']=='reserved' for f in value['workflows']))
        value['pricing_revisions'].append(deepcopy(pricing)); value['active_pricing']=pricing['id']
        b._validate_document(value)
        plan=_write_packet(b,previous['revision']+1,source,value,previous['scope_review_sha256'],previous_plan_sha256)
        _,_,cost,_=_chain(b,current=True,allow_pending=True)
    return Preview(_sha(_encode(plan)),plan['candidate_sha256'],str(cost))


def apply(b, *, approved_plan_sha256, approved_candidate_sha256, approve_application, writers_stopped):
    _require(approve_application is True and writers_stopped is True)
    _digest(approved_plan_sha256); _digest(approved_candidate_sha256)
    with b._physical_lock():
        plan,candidate,_,directory=_chain(b,current=True,allow_pending=True)
        _require(_sha(_encode(plan))==approved_plan_sha256 and plan['candidate_sha256']==approved_candidate_sha256
                 and _sha(b._read_bytes())==plan['source_sha256'] and _identity(b.path)==plan['source_ledger_identity']
                 and not os.path.lexists(directory/'applied.json'))
        initial=plan['revision']==0
        _require(os.path.lexists(b.directory/PREPARING)==initial)
        if initial:
            _require(_read_private(b,b.directory/PREPARING)==b'local-development-inert-v1'
                     and not os.path.lexists(b.directory/ACTIVATION))
        # Retain a global guard across ledger, approval receipt, activation and readback.
        b._mark_settlement()
        try:
            b._write(candidate)
            _require(b._read_bytes()==_encode(candidate))
            _write_private(b,directory/'applied.json',_receipt(plan))
            if initial: _write_private(b,b.directory/ACTIVATION,_receipt(plan))
            _chain(b,current=True)
            if initial:
                (b.directory/PREPARING).unlink(); _sync(b.directory)
            b.pending_path.unlink(); _sync(b.directory)
        except BaseException:
            b._blocked=True
            if not os.path.lexists(b.pending_path):
                try: b._mark_settlement()
                except (OSError,ValueError): pass
            raise


def _selection():
    return tuple(os.environ.get(n) for n in (
        ENV, PLAN_ENV, RUN_ENV, 'NORTHSTAR_LOCAL_TEST_PLAN', 'NORTHSTAR_LOCAL_TEST_RUN_APPROVAL',
        'NORTHSTAR_AUTHORITY_HANDOFF_PLAN', 'NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL',
        'NORTHSTAR_DEMO_BATCH_ENABLED', 'NORTHSTAR_BUDGET_AUTHORITY_ROOT',
        'NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED', 'NORTHSTAR_BUDGET_SCOPE', 'NORTHSTAR_BILLING_SCOPE',
        'NORTHSTAR_BUDGET_MAX_ATTEMPTS', 'NORTHSTAR_BUDGET_MAX_RESERVED_USD'))


def configured_development(root):
    from .live_access import local_backend, operator_access_authorized
    from .one_shot import one_shot_selected
    _require(os.environ.get(ENV)=='1' and local_backend()=='chroma' and operator_access_authorized()
             and not one_shot_selected() and os.environ.get('NORTHSTAR_DEMO_BATCH_ENABLED','0')=='0'
             and os.environ.get('NORTHSTAR_UI_ENABLE_GRAPH','0')=='0'
             and not any(k in os.environ for k in ('NORTHSTAR_BUDGET_AUTHORITY_ROOT',
                  'NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED','NORTHSTAR_BUDGET_SCOPE','NORTHSTAR_BILLING_SCOPE',
                  'NORTHSTAR_BUDGET_MAX_ATTEMPTS','NORTHSTAR_BUDGET_MAX_RESERVED_USD')))
    approval=os.environ.get(PLAN_ENV); _digest(approval)
    _require(os.environ.get(RUN_ENV)==run_approval(approval))
    b=open_review(root,approval=approval,run_approval=os.environ[RUN_ENV])
    b._selection=_selection()
    return b


def _review_price(root, relative):
    """Only an explicitly selected private pricing artifact, never .env/config."""
    path=contained_path(root,relative)
    _require(path.is_relative_to(root/'.local') and path.name.endswith('.json'))
    info=path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_uid==os.geteuid()
             and stat.S_IMODE(info.st_mode)==0o600 and info.st_nlink==1 and info.st_size<=65536)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        current=os.fstat(stream.fileno())
        _require((current.st_dev,current.st_ino)==(info.st_dev,info.st_ino))
        return _decode(stream.read(65537))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    for action in ('prepare','prepare-renewal','apply'):
        cmd=commands.add_parser(action)
        cmd.add_argument('--writers-stopped',action='store_true',required=True)
        if action=='apply':
            for name in ('approved-plan-sha256','approved-candidate-sha256'):
                cmd.add_argument('--'+name,required=True)
            cmd.add_argument('--approve-application',action='store_true',required=True)
        else:
            for name in ('pricing-file','verified-pricing-sha256',
                         'scope-review-sha256' if action=='prepare' else 'previous-plan-sha256'):
                cmd.add_argument('--'+name,required=True)
            cmd.add_argument('--approve-preparation',action='store_true',required=True)
    args=parser.parse_args(argv)
    try:
        root=validate_project_root(Path.cwd())
        if args.action=='apply':
            apply(open_review(root),approved_plan_sha256=args.approved_plan_sha256,
                  approved_candidate_sha256=args.approved_candidate_sha256,
                  approve_application=args.approve_application,writers_stopped=args.writers_stopped)
            print('Approved development candidate applied. Separate local execution approval is required.')
        else:
            price=_review_price(root,args.pricing_file)
            common=dict(verified_pricing_sha256=args.verified_pricing_sha256,
                        approve_preparation=args.approve_preparation,writers_stopped=args.writers_stopped)
            if args.action=='prepare':
                result=prepare(root,price,scope_review_sha256=args.scope_review_sha256,**common)
            else:
                result=prepare_renewal(open_review(root),price,previous_plan_sha256=args.previous_plan_sha256,**common)
            print(_encode(asdict(result)).decode())
        return 0
    except (OSError,ValueError,TypeError,KeyError,AttributeError,RecursionError):
        print('Development authority denied or incomplete. Stop; do not retry, reset or restore accounting.')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
