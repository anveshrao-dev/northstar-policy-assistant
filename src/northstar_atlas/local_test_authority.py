"""Explicit fresh, one-question Mac authority; never migrates or resets history.

Preparation is operator-only and leaves a durable guard. Activation and inference
are separate approvals. No runtime path creates accounting. Independent review
of scope uniqueness and total provider-account exposure is an operator duty;
local POSIX locking cannot discover or fence other hosts or arbitrary clients.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import stat
import uuid

from .budget_transition import _decode, _encode, _read_private, _sync, _write_private
from .diagnostic_window import WINDOW_NAME, digest, identity, sha
from .live_budget import (BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest,
                          LINEAR, _estimate, _fields, _require)
from .llm import MODEL_ID
from .project import contained_path, validate_project_root

KIND = 'v2-fresh-local-one-question'
POLICY = BudgetPolicy(1, Decimal('0.010000'))
PLAN = 'local-test-plan.json'
CANDIDATE = 'initial-candidate.json'
GUARD = b'fresh-local-preparation-v1'
PRICING_DIRECTORY = 'local-pricing-successor'
ACTIVE_PRICING_DIRECTORY = 'local-active-pricing-successor'
ENV_PLAN = 'NORTHSTAR_LOCAL_TEST_PLAN'
ENV_RUN = 'NORTHSTAR_LOCAL_TEST_RUN_APPROVAL'


@contextmanager
def _operation():
    try:
        yield
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise BudgetError('local test authority denied or incomplete; stop for operator review') from None


def _contract(request, corpus):
    from .local_embeddings import MODEL_ID as EMBEDDING, MODEL_REVISION
    _require(isinstance(request, WorkflowRequest) and request.profile == LINEAR and request.backend == 'chroma'
             and len(request.question) <= 2000)
    return dict(invocation_id=request.invocation_id, question_sha256=sha(request.question.encode()),
                mode=request.mode, backend='chroma', profile=LINEAR,
                reference_date=request.reference_date.isoformat(), model=MODEL_ID,
                corpus_sha256=corpus, embedding_model=EMBEDDING, embedding_revision=MODEL_REVISION)


def _identities(budget):
    return {name:identity(path) for name, path in (
        ('authority_identity', budget.root), ('local_identity', budget.root/'.local'),
        ('directory_identity', budget.directory), ('lock_identity', budget.lock_path))}


def _create(path, raw):
    """Create-only initial files. A partial preparation is never overwritten."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    _sync(path.parent)


@dataclass(frozen=True)
class Preview:
    plan_sha256: str
    candidate_sha256: str
    scope: str
    billing_scope: str
    reserved_usd: str


def prepare(root, request, pricing, *, verified_pricing_sha256, scope_review_sha256, approve_new_scope):
    """Explicit new-authority preparation only; existing directories are refused unread."""
    with _operation():
        root = validate_project_root(root)
        _require(approve_new_scope is True)
        digest(verified_pricing_sha256); digest(scope_review_sha256)
        _require(sha(_encode(pricing)) == verified_pricing_sha256)
        directory = contained_path(root, '.local/controlled-live')
        _require(not os.path.lexists(directory))
        for parent in (root, root/'.local'):
            info = parent.lstat()
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
                     and not stat.S_IMODE(info.st_mode) & 0o7022)
        corpus = sha(contained_path(root, 'data/corpus_manifest.json').read_bytes())
        contract = _contract(request, corpus)
        scope = uuid.uuid4().hex
        billing = 'local-test-' + scope
        budget = PersistentBudget(root, scope=scope, billing_scope=billing, policy=POLICY)
        value = dict(version=2, scope=scope, billing_scope=billing, provider='nebius', model=MODEL_ID,
            max_attempts=1, max_reserved_usd='0.010000', legacy=None, pricing_revisions=[pricing],
            active_pricing=pricing['id'], workflows=[], diagnostic_request_sha256=sha(_encode(contract)))
        _, prices, _ = budget._validate_document(value)
        cost = _estimate(prices[value['active_pricing']], 30720, 2048)
        _require(cost <= POLICY.max_reserved_usd)
        candidate = _encode(value)
        # mkdir is the create-only claim. No existing ledger, even a corrupt one,
        # is opened. A partial directory is a permanent stop, not a retry target.
        directory.mkdir(mode=0o700)
        _sync(directory.parent)
        _create(budget.pending_path, GUARD)
        _create(budget.lock_path, b'')
        _create(budget.path, candidate)
        budget._paths()
        _write_private(budget, directory/CANDIDATE, candidate)
        plan = dict(version=1, kind=KIND, authority=str(root), **_identities(budget),
            scope=scope, billing_scope=billing, new_cap=1, cost_cap='0.010000',
            candidate_sha256=sha(candidate), pricing_sha256=verified_pricing_sha256,
            scope_review_sha256=scope_review_sha256, request=contract)
        validate_plan(budget, plan)
        raw = _encode(plan)
        _write_private(budget, directory/PLAN, raw)
        return Preview(sha(raw), sha(candidate), scope, billing, str(cost))


def validate_plan(budget, plan):
    """Live validation never accepts expired pricing or a superseded approval."""
    if (os.path.lexists(budget.directory/ACTIVE_PRICING_DIRECTORY)
            or (isinstance(plan, dict) and plan.get('version') == 3)):
        from .local_active_pricing import validate_applied
        validate_applied(budget, plan)
        return
    if isinstance(plan, dict) and type(plan.get('version')) is int and plan['version'] == 2:
        from .local_test_pricing import validate_applied
        validate_applied(budget, plan)
        return
    _require(not os.path.lexists(budget.directory/PRICING_DIRECTORY))
    _validate_initial(budget, plan)


def _validate_initial(budget, plan, *, current=True):
    """Validate the approved root, complete identities, corpus and immutable seed."""
    from .local_embeddings import MODEL_ID as EMBEDDING, MODEL_REVISION
    _fields(plan, 'version kind authority authority_identity local_identity directory_identity lock_identity '
                  'scope billing_scope new_cap cost_cap candidate_sha256 pricing_sha256 scope_review_sha256 request')
    _require(type(plan['version']) is int and plan['version'] == 1 and plan['kind'] == KIND
             and budget.authority_root == budget.root and plan['authority'] == str(budget.root)
             and budget.policy == POLICY and plan['scope'] == budget.scope
             and plan['billing_scope'] == budget.billing_scope == 'local-test-'+budget.scope
             and type(plan['new_cap']) is int and plan['new_cap'] == 1 and plan['cost_cap'] == '0.010000')
    budget._paths()
    for name, pair in _identities(budget).items():
        _require(isinstance(plan[name], list) and len(plan[name]) == 2
                 and all(type(n) is int for n in plan[name]) and plan[name] == pair)
    for name in ('candidate_sha256', 'pricing_sha256', 'scope_review_sha256'): digest(plan[name])
    c = plan['request']
    _fields(c, 'invocation_id question_sha256 mode backend profile reference_date model corpus_sha256 embedding_model embedding_revision')
    probe = WorkflowRequest(c['invocation_id'], c['profile'], 'validation placeholder', c['mode'],
                            c['backend'], date.fromisoformat(c['reference_date']))
    digest(c['question_sha256'])
    validate_project_root(budget.root)
    corpus = sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes())
    expected = _contract(probe, corpus); expected['question_sha256'] = c['question_sha256']
    _require(c == expected and c['embedding_model'] == EMBEDDING and c['embedding_revision'] == MODEL_REVISION)
    seed = _read_private(budget, budget.directory/CANDIDATE)
    _require(sha(seed) == plan['candidate_sha256'])
    initial, prices, state = budget._validate_document(_decode(seed), current=current)
    _require(initial['legacy'] is None and initial['workflows'] == []
             and len(initial['pricing_revisions']) == 1
             and sha(_encode(initial['pricing_revisions'][0])) == plan['pricing_sha256']
             and initial['diagnostic_request_sha256'] == sha(_encode(c))
             and _estimate(prices[initial['active_pricing']], 30720, 2048) <= POLICY.max_reserved_usd)
    return seed


def _candidate_bytes(budget, plan):
    path = budget.directory/CANDIDATE
    if plan['version'] == 3:
        from .local_active_pricing import CANDIDATE as successor_candidate
        path = budget.directory/ACTIVE_PRICING_DIRECTORY/successor_candidate
    elif plan['version'] == 2:
        from .local_test_pricing import CANDIDATE as successor_candidate
        path = budget.directory/PRICING_DIRECTORY/successor_candidate
    return _read_private(budget, path)


def apply(budget, *, approved_plan_sha256, approved_candidate_sha256, writers_stopped, approve_activation):
    """Activate exactly the inert candidate under its original lock; no spending."""
    with _operation():
        _require(writers_stopped is True and approve_activation is True)
        digest(approved_plan_sha256); digest(approved_candidate_sha256)
        budget._paths()
        # The normal runtime lock deliberately rejects this preparation guard.
        # Only this create-only activation may hold the same inode while it exists.
        with os.fdopen(os.open(budget.lock_path, os.O_RDWR | os.O_NOFOLLOW), 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget._paths(); budget._opened_file(lock.fileno(), budget.lock_path)
            path = budget.directory/PLAN
            if os.path.lexists(budget.directory/PRICING_DIRECTORY):
                from .local_test_pricing import PLAN as successor_plan
                path = budget.directory/PRICING_DIRECTORY/successor_plan
            raw = _read_private(budget, path)
            _require(sha(raw) == approved_plan_sha256)
            plan = _decode(raw); validate_plan(budget, plan)
            _require(approved_candidate_sha256 == plan['candidate_sha256']
                     and sha(budget._read_bytes()) == approved_candidate_sha256
                     and _read_private(budget, budget.pending_path) == GUARD
                     and not os.path.lexists(budget.directory/WINDOW_NAME)
                     and not os.path.lexists(budget.directory/'authority-retired.json'))
            _write_private(budget, budget.directory/WINDOW_NAME, raw)
            try:
                # Readback and fsync precede removal; any uncertain completion
                # restores the guard where possible. No automatic rollback.
                budget._paths(); validate_plan(budget, plan)
                _require(sha(budget._read_bytes()) == approved_candidate_sha256)
                budget._sync_directory()
                budget.pending_path.unlink(); budget._sync_directory()
            except (OSError, ValueError):
                budget._blocked = True
                if not os.path.lexists(budget.pending_path):
                    try: _create(budget.pending_path, GUARD)
                    except (OSError, ValueError): pass
                raise


def selected():
    return ENV_PLAN in os.environ or ENV_RUN in os.environ


def run_approval(plan_id):
    digest(plan_id)
    return sha(('fresh-local-one-question-run:'+plan_id).encode())


def configured_plan():
    from .live_access import local_backend
    from .authority_handoff import one_shot_selected
    plan = os.environ.get(ENV_PLAN); digest(plan)
    _require(os.environ.get(ENV_RUN) == run_approval(plan) and not one_shot_selected()
             and local_backend() == 'chroma'
             and os.environ.get('NORTHSTAR_DEMO_BATCH_ENABLED', '0') == '0'
             and os.environ.get('NORTHSTAR_UI_ENABLE_GRAPH', '0') == '0'
             and not any(k in os.environ for k in ('NORTHSTAR_BUDGET_AUTHORITY_ROOT',
                                                   'NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED')))
    return plan


def check_request(budget, plan, request, corpus, ledger):
    from .live_access import require_local_access, require_local_request
    _require(configured_plan() == sha(_encode(plan)) and budget.before_request is require_local_access)
    require_local_request(budget.root, backend='chroma', graph=False)
    expected = _contract(request, corpus)
    _require(expected == plan['request'] and ledger['workflows'] == []
             and sha(budget._read_bytes()) == plan['candidate_sha256'])


@dataclass(frozen=True)
class InvocationStatus:
    plan_sha256: str
    invocation_id: str
    question_sha256: str
    mode: str
    reference_date: date
    available: bool
    backend: str = 'chroma'


def inspect_invocation(budget, plan_id, *, require_browser=True):
    from .live_access import require_live_budget, require_local_access, require_local_request
    with _operation():
        _require(plan_id == configured_plan() and budget.before_request is require_local_access)
        require_live_budget(budget.root, budget)
        if require_browser: require_local_request(budget.root, backend='chroma', graph=False)
        with budget._locked(diagnostic_approval=plan_id) as plan:
            _require(plan['kind'] == KIND)
            ledger, _, _ = budget._read()
            seed = _decode(_candidate_bytes(budget, plan))
            _require(dict(ledger, workflows=[]) == seed and len(ledger['workflows']) <= 1)
            c = plan['request']
            if ledger['workflows']:
                flow = ledger['workflows'][0]
                _require(flow['id'] == c['invocation_id'] and flow['profile'] == LINEAR
                         and flow['backend'] == 'chroma' and flow['mode'] == c['mode']
                         and flow['price_id'] == seed['active_pricing'])
            claim = contained_path(budget.root, f'.local/linear-demo-results/{c["invocation_id"]}.json')
            return InvocationStatus(plan_id, c['invocation_id'], c['question_sha256'], c['mode'],
                date.fromisoformat(c['reference_date']), not ledger['workflows'] and not os.path.lexists(claim))


def run_question(budget, request, *, plan_sha256, approved_run_sha256, consent, opener, limits):
    from .local_live import run_question as execute
    _require(consent is True and approved_run_sha256 == run_approval(plan_sha256)
             and inspect_invocation(budget, plan_sha256).available)
    return execute(budget, request, opener=opener, limits=limits, diagnostic_approval=plan_sha256)


def main(argv=None):
    """No default action. All commands operate only on the selected checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    prep = subs.add_parser('prepare')
    for flag in ('question-file', 'pricing-file', 'verified-pricing-sha256', 'scope-review-sha256',
                 'invocation-id', 'mode', 'reference-date'): prep.add_argument('--'+flag, required=True)
    prep.add_argument('--approve-new-scope', action='store_true', required=True)
    activate = subs.add_parser('apply')
    for flag in ('scope', 'billing-scope', 'approved-plan-sha256', 'approved-candidate-sha256'):
        activate.add_argument('--'+flag, required=True)
    activate.add_argument('--writers-stopped', action='store_true', required=True)
    activate.add_argument('--approve-activation', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        root = validate_project_root(Path.cwd())
        if args.action == 'prepare':
            probe = PersistentBudget(root)
            def read(relative):
                path = contained_path(root, relative)
                _require(path.is_relative_to(root/'.local'))
                return _read_private(probe, path)
            spec = WorkflowRequest(args.invocation_id, LINEAR, read(args.question_file).decode(), args.mode,
                                   'chroma', date.fromisoformat(args.reference_date))
            preview = prepare(root, spec, _decode(read(args.pricing_file)),
                verified_pricing_sha256=args.verified_pricing_sha256, scope_review_sha256=args.scope_review_sha256,
                approve_new_scope=args.approve_new_scope)
            from dataclasses import asdict
            print(_encode(dict(asdict(preview), run_approval_sha256=run_approval(preview.plan_sha256))).decode())
        else:
            budget = PersistentBudget(root, scope=args.scope, billing_scope=args.billing_scope, policy=POLICY)
            apply(budget, approved_plan_sha256=args.approved_plan_sha256,
                  approved_candidate_sha256=args.approved_candidate_sha256,
                  writers_stopped=args.writers_stopped, approve_activation=args.approve_activation)
            print('Approved local candidate activated; inference requires separate run approval and session consent.')
        return 0
    except (OSError, ValueError):
        print('Local authority operation denied or incomplete. Stop; do not retry, delete or restore accounting.')
        return 1


if __name__ == '__main__': raise SystemExit(main())
