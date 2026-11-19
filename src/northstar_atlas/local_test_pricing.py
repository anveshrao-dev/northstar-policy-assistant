"""One pricing successor for an expired, inactive local one-question candidate.

Preparation, application and activation are separate approvals. This module
never activates an authority, initializes a ledger or contacts a provider.
Original plans, candidates and pricing remain immutable historical evidence.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import os
from pathlib import Path

from . import local_test_authority as local
from .budget_transition import _decode, _encode, _read_private, _write_private, _private_directory, _sync
from .diagnostic_window import WINDOW_NAME, digest, sha
from .live_budget import PersistentBudget, _require, _timestamp, _estimate
from .project import contained_path, validate_project_root

PLAN = 'plan.json'
CANDIDATE = 'candidate.json'
INTENT = 'application-intent.json'
PENDING = 'application.pending'
APPLIED = 'applied.json'


def _directory(budget):
    return contained_path(budget.authority_root, '.local/controlled-live/'+local.PRICING_DIRECTORY)


@contextmanager
def _inactive_lock(budget, writers_stopped):
    """The original exclusive lock, allowing only the exact preparation guard."""
    with local._operation():
        _require(writers_stopped is True and budget.policy == local.POLICY
                 and budget.authority_root == budget.root and not budget._blocked)
        budget._paths()
        with os.fdopen(os.open(budget.lock_path, os.O_RDWR | os.O_NOFOLLOW), 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget._paths(); budget._opened_file(lock.fileno(), budget.lock_path)
            _require(_read_private(budget, budget.pending_path) == local.GUARD
                     and not os.path.lexists(budget.directory/WINDOW_NAME)
                     and not os.path.lexists(budget.directory/'authority-retired.json'))
            original = _decode(_read_private(budget, budget.directory/local.PLAN))
            invocation = original['request']['invocation_id']
            claim = contained_path(budget.root, '.local/linear-demo-results/'+invocation+'.json')
            _require(not os.path.lexists(claim))
            yield


def _original(budget, previous_plan_sha256):
    digest(previous_plan_sha256)
    raw = _read_private(budget, budget.directory/local.PLAN)
    _require(sha(raw) == previous_plan_sha256)
    original = _decode(raw)
    seed = local._validate_initial(budget, original, current=False)
    return original, seed


def _candidate(budget, seed, pricing, *, current=True):
    old = _decode(seed)
    previous = old['pricing_revisions'][0]
    _require(_timestamp(previous['valid_until']) <= budget.clock()
             and _timestamp(pricing['confirmed_at']) >= _timestamp(previous['valid_until']))
    candidate = deepcopy(old)
    candidate['pricing_revisions'].append(deepcopy(pricing))
    candidate['active_pricing'] = pricing['id']
    _, prices, state = budget._validate_document(candidate, current=current)
    _require(state.attempts_consumed == 0 and state.reserved_usd == 0)
    cost = _estimate(prices[candidate['active_pricing']], 30720, 2048)
    _require(cost <= local.POLICY.max_reserved_usd)
    return _encode(candidate), cost


def _validate_successor(budget, plan, *, current=True):
    """Only the historical parent may have expired; the successor must be current."""
    _require(isinstance(plan, dict) and type(plan.get('version')) is int and plan['version'] == 2)
    original, seed = _original(budget, plan['previous_plan_sha256'])
    directory = _directory(budget); _private_directory(budget, directory)
    raw = _read_private(budget, directory/CANDIDATE)
    value = _decode(raw)
    _require(isinstance(value.get('pricing_revisions'), list) and len(value['pricing_revisions']) == 2)
    pricing = value['pricing_revisions'][-1]
    expected, cost = _candidate(budget, seed, pricing, current=current)
    _require(raw == expected)
    expected_plan = dict(original, version=2, previous_plan_sha256=sha(_encode(original)),
        previous_candidate_sha256=sha(seed), candidate_sha256=sha(raw), pricing_sha256=sha(_encode(pricing)))
    # Bind the exact preserved bytes, not a reformatted original document.
    expected_plan['previous_plan_sha256'] = sha(_read_private(budget, budget.directory/local.PLAN))
    _require(plan == expected_plan and _read_private(budget, directory/PLAN) == _encode(plan))
    return raw, seed, cost


def _receipt(plan):
    return _encode(dict(plan_sha256=sha(_encode(plan)), candidate_sha256=plan['candidate_sha256'],
                        previous_candidate_sha256=plan['previous_candidate_sha256']))


def validate_applied(budget, plan, *, current=True):
    # Historical-only inspection is used by the activated renewal parent check.
    # Every existing activation/runtime caller retains the current=True default.
    _validate_successor(budget, plan, current=current)
    directory = _directory(budget)
    _require(not os.path.lexists(directory/PENDING)
             and _read_private(budget, directory/INTENT) == _receipt(plan)
             and _read_private(budget, directory/APPLIED) == _receipt(plan))


def prepare(budget, pricing, *, previous_plan_sha256, verified_pricing_sha256,
            writers_stopped, approve_preparation):
    """Create one inert successor; any partial directory blocks old activation."""
    with _inactive_lock(budget, writers_stopped):
        _require(approve_preparation is True)
        digest(verified_pricing_sha256)
        _require(sha(_encode(pricing)) == verified_pricing_sha256)
        directory = _directory(budget)
        _require(not os.path.lexists(directory))
        original, seed = _original(budget, previous_plan_sha256)
        _require(budget._read_bytes() == seed)
        candidate, cost = _candidate(budget, seed, pricing)
        plan = dict(original, version=2, previous_plan_sha256=previous_plan_sha256,
            previous_candidate_sha256=sha(seed), candidate_sha256=sha(candidate),
            pricing_sha256=verified_pricing_sha256)
        directory.mkdir(mode=0o700); _sync(budget.directory)
        _write_private(budget, directory/CANDIDATE, candidate)
        _write_private(budget, directory/PLAN, _encode(plan))
        _validate_successor(budget, plan)
        _require(budget._read_bytes() == seed)
        return local.Preview(sha(_encode(plan)), sha(candidate), budget.scope, budget.billing_scope, str(cost))


def apply(budget, *, approved_plan_sha256, approved_candidate_sha256,
          writers_stopped, approve_application):
    """Install approved pricing under the guard. Never activates or refunds."""
    with _inactive_lock(budget, writers_stopped):
        _require(approve_application is True)
        digest(approved_plan_sha256); digest(approved_candidate_sha256)
        directory = _directory(budget); _private_directory(budget, directory)
        raw = _read_private(budget, directory/PLAN)
        _require(sha(raw) == approved_plan_sha256)
        plan = _decode(raw)
        candidate, seed, _ = _validate_successor(budget, plan)
        _require(sha(candidate) == approved_candidate_sha256 and budget._read_bytes() == seed
                 and not any(os.path.lexists(directory/name) for name in (PENDING, INTENT, APPLIED)))
        # Persistent intent rejects automatic retry even if replacement never occurred.
        _write_private(budget, directory/PENDING, _receipt(plan))
        try:
            _write_private(budget, directory/INTENT, _receipt(plan))
            budget._write(candidate)
            _require(budget._read_bytes() == candidate)
            _validate_successor(budget, plan)
            budget._sync_directory()
            _write_private(budget, directory/APPLIED, _receipt(plan))
            (directory/PENDING).unlink(); _sync(directory)
        except (OSError, ValueError):
            budget._blocked = True
            if not os.path.lexists(directory/PENDING):
                try: _write_private(budget, directory/PENDING, _receipt(plan))
                except (OSError, ValueError): pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    for action in ('prepare', 'apply'):
        command = subs.add_parser(action)
        for field in ('scope', 'billing-scope'): command.add_argument('--'+field, required=True)
        command.add_argument('--writers-stopped', action='store_true', required=True)
        if action == 'prepare':
            for field in ('pricing-file', 'previous-plan-sha256', 'verified-pricing-sha256'):
                command.add_argument('--'+field, required=True)
            command.add_argument('--approve-preparation', action='store_true', required=True)
        else:
            for field in ('approved-plan-sha256', 'approved-candidate-sha256'):
                command.add_argument('--'+field, required=True)
            command.add_argument('--approve-application', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        root = validate_project_root(Path.cwd())
        budget = PersistentBudget(root, scope=args.scope, billing_scope=args.billing_scope, policy=local.POLICY)
        if args.action == 'prepare':
            path = contained_path(root, args.pricing_file)
            _require(path.is_relative_to(root/'.local'))
            preview = prepare(budget, _decode(_read_private(budget, path)),
                previous_plan_sha256=args.previous_plan_sha256,
                verified_pricing_sha256=args.verified_pricing_sha256,
                writers_stopped=args.writers_stopped, approve_preparation=args.approve_preparation)
            print(_encode(asdict(preview)).decode())
        else:
            apply(budget, approved_plan_sha256=args.approved_plan_sha256,
                approved_candidate_sha256=args.approved_candidate_sha256,
                writers_stopped=args.writers_stopped, approve_application=args.approve_application)
            print('Pricing applied; authority remains inert. Separate activation and inference approvals required.')
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        print('Local pricing denied or incomplete; stop for review. Do not retry or restore accounting.')
        return 1


if __name__ == '__main__': raise SystemExit(main())
