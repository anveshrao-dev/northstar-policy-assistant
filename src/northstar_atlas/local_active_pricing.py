"""One pricing renewal for an activated, unused local one-question authority.

The original activation remains immutable. A separately approved successor is
resolved under the same accounting lock; it never creates spending capacity.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path

from . import local_test_authority as local
from .budget_transition import _decode, _encode, _private_directory, _read_private, _sync, _write_private
from .diagnostic_window import WINDOW_NAME, digest, sha
from .live_budget import PersistentBudget, _estimate, _require, _timestamp
from .project import contained_path, validate_project_root

PLAN = 'plan.json'
CANDIDATE = 'candidate.json'
SOURCE = 'source.json'
ACTIVATION = 'activation.json'
PENDING = 'application.pending'
INTENT = 'application-intent.json'
APPLIED = 'applied.json'
GUARD = b'active-local-pricing-renewal-v1'


def _directory(budget):
    return contained_path(budget.authority_root,
                          '.local/controlled-live/'+local.ACTIVE_PRICING_DIRECTORY)


def _parent(budget):
    """Historical checks only; never authorize execution of the expired parent."""
    _require(budget.policy == local.POLICY and budget.authority_root == budget.root)
    raw = _read_private(budget, budget.directory/WINDOW_NAME)
    parent = _decode(raw)
    _require(isinstance(parent, dict) and type(parent.get('version')) is int
             and parent['version'] in (1, 2) and parent.get('kind') == local.KIND)
    if parent['version'] == 1:
        source = local._validate_initial(budget, parent, current=False)
        _require(raw == _read_private(budget, budget.directory/local.PLAN))
    else:
        from . import local_test_pricing as inert
        inert.validate_applied(budget, parent, current=False)
        directory = budget.directory/local.PRICING_DIRECTORY
        _require(raw == _read_private(budget, directory/inert.PLAN))
        source = _read_private(budget, directory/inert.CANDIDATE)
    _require(sha(source) == parent['candidate_sha256'])
    value, _, state = budget._validate_document(_decode(source), current=False)
    _require(value['workflows'] == [] and state.attempts_consumed == 0 and state.reserved_usd == 0)
    return raw, parent, source


def _candidate(budget, source, pricing):
    old, prices, _ = budget._validate_document(_decode(source), current=False)
    expiry = _timestamp(prices[old['active_pricing']]['valid_until'])
    _require(expiry <= budget.clock() and _timestamp(pricing['confirmed_at']) >= expiry)
    candidate = deepcopy(old)
    candidate['pricing_revisions'].append(deepcopy(pricing))
    candidate['active_pricing'] = pricing['id']
    _, current_prices, state = budget._validate_document(candidate)
    _require(candidate['workflows'] == [] and state.attempts_consumed == 0 and state.reserved_usd == 0)
    cost = _estimate(current_prices[candidate['active_pricing']], 30720, 2048)
    _require(cost <= local.POLICY.max_reserved_usd)
    return _encode(candidate), cost


@contextmanager
def _unused_lock(budget, writers_stopped):
    with local._operation():
        _require(writers_stopped is True)
        with budget._physical_lock():
            raw, parent, source = _parent(budget)
            _require(budget._read_bytes() == source)
            claim = contained_path(budget.root,
                '.local/linear-demo-results/'+parent['request']['invocation_id']+'.json')
            _require(not os.path.lexists(claim))
            yield raw, parent, source


def _validate_successor(budget, plan):
    _require(isinstance(plan, dict) and type(plan.get('version')) is int and plan['version'] == 3)
    directory = _directory(budget); _private_directory(budget, directory)
    activation, parent, source = _parent(budget)
    _require(_read_private(budget, directory/SOURCE) == source
             and _read_private(budget, directory/ACTIVATION) == activation)
    raw = _read_private(budget, directory/CANDIDATE)
    candidate = _decode(raw)
    _require(isinstance(candidate, dict) and isinstance(candidate.get('pricing_revisions'), list)
             and len(candidate['pricing_revisions']) == len(_decode(source)['pricing_revisions'])+1)
    pricing = candidate['pricing_revisions'][-1]
    expected, cost = _candidate(budget, source, pricing)
    identity = plan.get('source_ledger_identity')
    _require(isinstance(identity, list) and len(identity) == 2
             and all(type(n) is int and n >= 0 for n in identity))
    expected_plan = dict(parent, version=3, previous_activation_sha256=sha(activation),
        source_sha256=sha(source), source_ledger_identity=identity,
        candidate_sha256=sha(raw), pricing_sha256=sha(_encode(pricing)))
    _require(raw == expected and plan == expected_plan
             and _read_private(budget, directory/PLAN) == _encode(plan))
    return raw, cost


def _receipt(plan):
    return _encode(dict(plan_sha256=sha(_encode(plan)), candidate_sha256=plan['candidate_sha256'],
                        source_sha256=plan['source_sha256'],
                        previous_activation_sha256=plan['previous_activation_sha256']))


def validate_applied(budget, plan):
    """Runtime validation: only the complete, current successor is executable."""
    _validate_successor(budget, plan)
    directory = _directory(budget)
    _require(not os.path.lexists(budget.pending_path) and not os.path.lexists(directory/PENDING)
             and _read_private(budget, directory/INTENT) == _receipt(plan)
             and _read_private(budget, directory/APPLIED) == _receipt(plan))


def resolve_window(budget, activation, approval):
    """The runtime lock is already held; retain the original activation bytes."""
    raw = _read_private(budget, _directory(budget)/PLAN)
    _require(sha(raw) == approval)
    plan = _decode(raw)
    validate_applied(budget, plan)
    _require(sha(activation) == plan['previous_activation_sha256']
             and budget.policy.max_attempts == plan['new_cap'])
    return plan


def prepare(budget, pricing, *, previous_activation_sha256, verified_pricing_sha256,
            writers_stopped, approve_preparation):
    """Prepare once. Even an incomplete directory fences historical approvals."""
    with _unused_lock(budget, writers_stopped) as (activation, parent, source):
        _require(approve_preparation is True)
        digest(previous_activation_sha256); digest(verified_pricing_sha256)
        _require(sha(activation) == previous_activation_sha256
                 and sha(_encode(pricing)) == verified_pricing_sha256)
        directory = _directory(budget)
        _require(not os.path.lexists(directory))
        candidate, cost = _candidate(budget, source, pricing)
        info = budget.path.lstat()
        plan = dict(parent, version=3, previous_activation_sha256=sha(activation),
            source_sha256=sha(source), source_ledger_identity=[info.st_dev, info.st_ino],
            candidate_sha256=sha(candidate), pricing_sha256=verified_pricing_sha256)
        directory.mkdir(mode=0o700); _sync(budget.directory)
        for name, raw in ((SOURCE, source), (ACTIVATION, activation),
                          (CANDIDATE, candidate), (PLAN, _encode(plan))):
            _write_private(budget, directory/name, raw)
        _validate_successor(budget, plan)
        _require(budget._read_bytes() == source)
        return local.Preview(sha(_encode(plan)), sha(candidate), budget.scope, budget.billing_scope, str(cost))


def apply(budget, *, approved_plan_sha256, approved_candidate_sha256,
          writers_stopped, approve_application):
    """Apply once under two durable fences. No retry, rollback or new capacity."""
    with _unused_lock(budget, writers_stopped) as (_, _, source):
        _require(approve_application is True)
        digest(approved_plan_sha256); digest(approved_candidate_sha256)
        directory = _directory(budget); _private_directory(budget, directory)
        raw = _read_private(budget, directory/PLAN)
        _require(sha(raw) == approved_plan_sha256)
        plan = _decode(raw); candidate, _ = _validate_successor(budget, plan)
        info = budget.path.lstat()
        _require(sha(candidate) == approved_candidate_sha256 and sha(source) == plan['source_sha256']
                 and [info.st_dev, info.st_ino] == plan['source_ledger_identity']
                 and not any(os.path.lexists(directory/name) for name in (PENDING, INTENT, APPLIED)))
        _write_private(budget, budget.pending_path, GUARD)
        try:
            _write_private(budget, directory/PENDING, _receipt(plan))
            _write_private(budget, directory/INTENT, _receipt(plan))
            budget._write(candidate)
            _require(budget._read_bytes() == candidate)
            _validate_successor(budget, plan); budget._sync_directory()
            _write_private(budget, directory/APPLIED, _receipt(plan))
            _require(_read_private(budget, directory/APPLIED) == _receipt(plan))
            (directory/PENDING).unlink(); _sync(directory)
            budget.pending_path.unlink(); budget._sync_directory()
        except (OSError, ValueError, KeyboardInterrupt, SystemExit):
            budget._blocked = True
            for path, contents in ((budget.pending_path, GUARD), (directory/PENDING, _receipt(plan))):
                if not os.path.lexists(path):
                    try: _write_private(budget, path, contents)
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
            for field in ('pricing-file', 'previous-activation-sha256', 'verified-pricing-sha256'):
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
                previous_activation_sha256=args.previous_activation_sha256,
                verified_pricing_sha256=args.verified_pricing_sha256,
                writers_stopped=args.writers_stopped, approve_preparation=args.approve_preparation)
            print(_encode(asdict(preview)).decode())
        else:
            apply(budget, approved_plan_sha256=args.approved_plan_sha256,
                approved_candidate_sha256=args.approved_candidate_sha256,
                writers_stopped=args.writers_stopped, approve_application=args.approve_application)
            print('Approved pricing applied; separate successor run approval and session consent are required.')
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        print('Active local pricing denied or incomplete. Stop; do not retry or restore accounting.')
        return 1


if __name__ == '__main__': raise SystemExit(main())
