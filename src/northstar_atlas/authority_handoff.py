"""Operator-reviewed, one-way continuity for an unused Standard RAG batch.

No initialization, transport, pricing change, automatic recovery or rollback.
Cross-host receipts require independent operator verification; a digest is not
remote process attestation. Only reviewed lock-cooperating writers are covered.
"""
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
from types import SimpleNamespace

from .budget_transition import (_encode, _decode, _read_private, _write_private,
                                _private_directory, _sync, TransitionPreview)
from .diagnostic_window import (WINDOW_NAME, DEMO_KIND, DEMO_BILLING_SCOPE,
                                digest, sha, validate_contract, _validate_plan_fields)
from .live_budget import BudgetError, BudgetPolicy, LINEAR, _fields, _require
from .llm import MODEL_ID
from .project import contained_path

KIND = 'v2-local-authority-successor'
RETIRED = 'authority-retired.json'
_FILES = ('source.v2.json', 'previous-window.json', 'candidate.v2.json', 'chain.json')


@contextmanager
def _operation():
    try:
        yield
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise BudgetError('authority handoff denied or uncertain; keep writers stopped') from None


def _identity(budget):
    budget._paths()
    result = {'root': str(budget.authority_root)}
    for name, path in (('authority', budget.authority_root), ('local', budget.authority_root/'.local'),
                       ('directory', budget.directory), ('lock', budget.lock_path), ('ledger', budget.path)):
        info = path.lstat()
        result[name] = dict(device=info.st_dev, inode=info.st_ino, owner=info.st_uid)
    return result


def _identity_shape(value):
    _fields(value, 'root authority local directory lock ledger')
    raw = value['root']; path = Path(raw)
    _require(isinstance(raw, str) and raw == raw.strip() and path.is_absolute()
             and str(path) == raw and '..' not in path.parts)
    for name in ('authority', 'local', 'directory', 'lock', 'ledger'):
        _fields(value[name], 'device inode owner')
        _require(all(type(n) is int and n >= 0 for n in value[name].values()))
    _require(len({value[name]['owner'] for name in ('authority','local','directory','lock','ledger')}) == 1)


def _matches(budget, expected, *, ledger=True):
    _identity_shape(expected)
    current = _identity(budget)
    keys = current.keys() if ledger else current.keys()-{'ledger'}
    _require(all(current[k] == expected[k] for k in keys))


def _deadline(budget, plan):
    start, end = (datetime.fromisoformat(plan[k]) for k in ('prepared_at', 'expires_at'))
    _require(start.tzinfo is not None and end.tzinfo is not None
             and start <= budget.clock() < end)


def _directory(budget, plan_id):
    digest(plan_id)
    return budget.directory/('authority-handoff-' + plan_id)


def describe_destination(budget):
    """Inspect an explicitly staged, inert copy. Never creates a destination."""
    with _operation(), budget._physical_lock():
        raw = budget._read_bytes(); window = _read_private(budget, budget.directory/WINDOW_NAME)
        previous = _decode(window)
        _require(budget.policy == BudgetPolicy(12) and previous['authority'] != str(budget.authority_root))
        budget._validate_document(_decode(raw))
        # Old identity-bound window cannot authorize this path; removing it also
        # fails because the ledger retains diagnostic_request_sha256.
        _require('diagnostic_request_sha256' in _decode(raw))
        return dict(identity=_identity(budget), source_sha256=sha(raw), previous_sha256=sha(window),
                    corpus_sha256=sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()))


def _archive(budget, window):
    """Capture only the reviewed approval chain, without altering its bytes."""
    rows = []
    prefixes = {'v2-unused-demo-batch-pricing-renewal':'demo-batch-pricing-',
                'v2-five-question-demo-batch':'demo-batch-', DEMO_KIND:'demo-amendment-'}
    for _ in range(4):
        plan = _decode(window)
        directory = budget.directory/(prefixes[plan['kind']] + sha(window))
        _private_directory(budget, directory)
        _require(_read_private(budget, directory/'plan.json') == window)
        rows.append(dict(window=window.decode(), source=_read_private(budget, directory/'backup.v2.json').decode(),
                         candidate=_read_private(budget, directory/'candidate.v2.json').decode()))
        if plan['kind'] == DEMO_KIND:
            return rows
        window = _read_private(budget, directory/'backup.window.json')
    raise BudgetError()


def _schema(budget, value, cap):
    validator = copy(budget); validator.policy = BudgetPolicy(cap)
    return validator._validate_document(value, current=False)


def _validate_chain(budget, plan, chain, source, window):
    """Historical validation never reinterprets old paths as current authority."""
    from .demo_batch import KIND as BATCH, _active_price
    from .demo_batch_pricing import KIND as RENEWAL
    _require(isinstance(chain, list) and len(chain) in (3, 4) and chain[0]['window'].encode() == window)
    historical = SimpleNamespace(authority_root=Path(plan['source_identity']['root']),
        scope=budget.scope, billing_scope=budget.billing_scope)
    parent_candidate = None
    for row in reversed(chain):
        _fields(row, 'window source candidate')
        approval = _decode(row['window'].encode()); before = _decode(row['source'].encode())
        after = _decode(row['candidate'].encode())
        _validate_plan_fields(historical, approval)
        _require(sha(row['source'].encode()) == approval['source_sha256']
                 and sha(row['candidate'].encode()) == approval['candidate_sha256'])
        for key, name in (('authority_identity','authority'), ('local_identity','local'),
                          ('directory_identity','directory'), ('lock_identity','lock')):
            item = plan['source_identity'][name]
            _require(approval[key] == [item['device'], item['inode']])
        _, _, old_state = _schema(budget, before, approval['old_cap'])
        _, _, new_state = _schema(budget, after, approval['new_cap'])
        _require(old_state == new_state)
        expected = deepcopy(before)
        expected.update(max_attempts=approval['new_cap'], diagnostic_request_sha256=sha(_encode(approval['request'])))
        if approval['kind'] in (DEMO_KIND, RENEWAL):
            if len(after['pricing_revisions']) == len(before['pricing_revisions'])+1:
                expected['pricing_revisions'].append(after['pricing_revisions'][-1])
                expected['active_pricing'] = after['active_pricing']
            elif approval['kind'] == RENEWAL:
                raise BudgetError()
        _require(after == expected and sha(_encode(_active_price(after))) == approval['pricing_sha256'])
        if parent_candidate is None:
            _require(approval['kind'] == DEMO_KIND and len(before['workflows']) == 1
                     and (before['legacy'] is None or before['legacy']['attempts'] == []))
        else:
            old_approval, parent = parent_candidate
            _require(approval['previous_window_sha256'] == sha(_encode(old_approval)))
            prefix = deepcopy(before); prefix['workflows'] = prefix['workflows'][:len(parent['workflows'])]
            _require(prefix == parent)
            tail = before['workflows'][len(parent['workflows']):]
            expected_count = 0 if approval['kind'] == RENEWAL else 1 if approval['old_cap'] == 2 else 5
            _require(len(tail) == expected_count)
            if approval['kind'] == RENEWAL:
                _require(old_approval['kind'] == BATCH and old_approval['old_cap'] == 7
                         and approval['request'] == old_approval['request'])
        parent_candidate = approval, after
    final, baseline = parent_candidate
    _require(final['kind'] in (BATCH, RENEWAL) and final['new_cap'] == 12
             and (final['old_cap'] == 7 if final['kind'] == BATCH else final['old_cap'] == 12)
             and baseline == _decode(source) and source == chain[0]['candidate'].encode()
             and len(baseline['workflows']) == 7
             and all(f['profile'] == LINEAR and f['backend'] == 'pinecone'
                     and f['status'] in ('closed', 'uncertain') for f in baseline['workflows']))


def validate_successor(budget, plan):
    _fields(plan, 'version kind source_identity destination_identity source_sha256 previous_sha256 '
        'candidate_sha256 chain_sha256 scope billing_scope model old_cap new_cap cost_cap pricing_sha256 '
        'prepared_at expires_at writer_evidence_sha256 request')
    _require(type(plan['version']) is int and plan['version'] == 7 and plan['kind'] == KIND
             and plan['scope'] == budget.scope and plan['billing_scope'] == budget.billing_scope == DEMO_BILLING_SCOPE
             and plan['model'] == MODEL_ID and type(plan['old_cap']) is int and plan['old_cap'] == 12
             and type(plan['new_cap']) is int and plan['new_cap'] == 12 and plan['cost_cap'] == '0.100000'
             and budget.policy == BudgetPolicy(12))
    for key in ('source_sha256','previous_sha256','candidate_sha256','chain_sha256','pricing_sha256','writer_evidence_sha256'):
        digest(plan[key])
    _identity_shape(plan['source_identity']); _identity_shape(plan['destination_identity'])
    source = Path(plan['source_identity']['root']); destination = Path(plan['destination_identity']['root'])
    _require(not source.is_relative_to(destination) and not destination.is_relative_to(source))
    validate_contract(plan['request'], profile=LINEAR)
    _deadline(budget, plan)


def _packet(budget, plan_id):
    directory = _directory(budget, plan_id); _private_directory(budget, directory)
    raw = _read_private(budget, directory/'plan.json'); _require(len(raw) <= 16384 and sha(raw) == plan_id)
    plan = _decode(raw); validate_successor(budget, plan)
    files = {name:_read_private(budget, directory/name) for name in _FILES}
    for name, field in zip(_FILES, ('source_sha256','previous_sha256','candidate_sha256','chain_sha256')):
        _require(sha(files[name]) == plan[field])
    source = files['source.v2.json']; candidate = files['candidate.v2.json']
    _validate_chain(budget, plan, _decode(files['chain.json']), source, files['previous-window.json'])
    old, _, state = budget._validate_document(_decode(source))
    expected = deepcopy(old); expected['diagnostic_request_sha256'] = sha(_encode(plan['request']))
    _require(candidate == _encode(expected) and budget._validate_document(_decode(candidate))[2] == state)
    from .demo_batch import _active_price, _cost
    price = _active_price(old)
    _require(sha(_encode(price)) == plan['pricing_sha256']
             and datetime.fromisoformat(plan['expires_at']) <= datetime.fromisoformat(price['valid_until'])
             and state.attempts_consumed < 12 and state.reserved_usd+_cost(price) <= budget.policy.max_reserved_usd
             and all(f['id'] != plan['request']['invocation_id'] for f in old['workflows'])
             and plan['request']['corpus_sha256'] == sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()))
    return directory, raw, plan, files, state


def prepare_handoff(budget, request, *, previous_plan_sha256, destination, approved_destination_sha256,
                    writers_stopped, reviewed_writer_evidence_sha256, expires_at, dry_run=True):
    """Prepare immutable artifacts only; concrete approval does not authorize inference."""
    with _operation(), budget._locked(diagnostic_approval=previous_plan_sha256) as previous:
        _require(writers_stopped is True and type(dry_run) is bool)
        digest(reviewed_writer_evidence_sha256); digest(approved_destination_sha256)
        _fields(destination, 'identity source_sha256 previous_sha256 corpus_sha256')
        _require(sha(_encode(destination)) == approved_destination_sha256)
        from .demo_batch import _progress, _active_price
        source = budget._read_bytes(); ledger = _decode(source)
        count, state, _ = _progress(budget, previous, ledger)
        _require(count == 5 and len(ledger['workflows']) == 7 and budget.policy == BudgetPolicy(12))
        window = _read_private(budget, budget.directory/WINDOW_NAME)
        corpus = sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes())
        contract = dict(invocation_id=request.invocation_id, question_sha256=sha(request.question.encode()),
            mode=request.mode, backend=request.backend, profile=request.profile,
            reference_date=request.reference_date.isoformat(), model=MODEL_ID, corpus_sha256=corpus)
        validate_contract(contract, profile=LINEAR)
        _require(destination['source_sha256'] == sha(source) and destination['previous_sha256'] == sha(window)
                 and destination['corpus_sha256'] == corpus and sha(window) == previous_plan_sha256)
        chain = _encode(_archive(budget, window))
        candidate = deepcopy(ledger); candidate['diagnostic_request_sha256'] = sha(_encode(contract))
        candidate_raw = _encode(candidate)
        plan = dict(version=7, kind=KIND, source_identity=_identity(budget), destination_identity=destination['identity'],
            source_sha256=sha(source), previous_sha256=sha(window), candidate_sha256=sha(candidate_raw), chain_sha256=sha(chain),
            scope=budget.scope, billing_scope=budget.billing_scope, model=MODEL_ID, old_cap=12, new_cap=12,
            cost_cap='0.100000', pricing_sha256=sha(_encode(_active_price(ledger))), prepared_at=budget.clock().isoformat(),
            expires_at=expires_at, writer_evidence_sha256=reviewed_writer_evidence_sha256, request=contract)
        validate_successor(budget, plan)
        _validate_chain(budget, plan, _decode(chain), source, window)
        _require(datetime.fromisoformat(expires_at) <= datetime.fromisoformat(_active_price(ledger)['valid_until'])
                 and all(f['id'] != request.invocation_id for f in ledger['workflows']))
        raw = _encode(plan); plan_id = sha(raw); directory = None
        if not dry_run:
            directory = _directory(budget, plan_id); directory.mkdir(mode=0o700); _sync(budget.directory)
            for name, data in zip(('plan.json', *_FILES), (raw, source, window, candidate_raw, chain)):
                _write_private(budget, directory/name, data)
            _packet(budget, plan_id)
        return TransitionPreview(plan_id, sha(candidate_raw), sha(source), state.attempts_consumed,
                                 str(state.reserved_usd), directory)


def _tombstone(plan_id, files):
    return _encode(dict(version='retired-authority-v1', plan_sha256=plan_id,
                        source_sha256=sha(files['source.v2.json']), history=_decode(files['source.v2.json'])))


def retire_source(budget, plan_id, *, approved_plan_sha256, writers_stopped, reviewed_writer_evidence_sha256):
    """Irreversible source fence. No rollback or automatic receipt recovery."""
    with _operation(), budget._physical_lock():
        _require(writers_stopped is True and approved_plan_sha256 == plan_id)
        directory, raw, plan, files, _ = _packet(budget, plan_id)
        _matches(budget, plan['source_identity'])
        _require(reviewed_writer_evidence_sha256 == plan['writer_evidence_sha256'])
        from .diagnostic_window import read_window
        previous = read_window(budget, plan['previous_sha256'])
        from .demo_batch import _progress
        _require(_progress(budget, previous, _decode(files['source.v2.json']))[0] == 5
                 and _encode(_archive(budget, files['previous-window.json'])) == files['chain.json'])

        def unchanged():
            _matches(budget, plan['source_identity']); _deadline(budget, plan)
            _require(budget._read_bytes() == files['source.v2.json']
                     and _read_private(budget, budget.directory/WINDOW_NAME) == files['previous-window.json'])
        unchanged()
        tombstone = _tombstone(plan_id, files)
        marker = _encode(dict(plan_sha256=plan_id, tombstone_sha256=sha(tombstone)))

        def fence():
            unchanged()
            _write_private(budget, budget.directory/RETIRED, marker)

        def verify():
            _require(budget._read_bytes() == tombstone
                     and _read_private(budget, budget.directory/RETIRED) == marker)
            _matches(budget, plan['source_identity'], ledger=False)
        budget._settle(tombstone, before_replace=fence, verify=verify)
        # Receipt publication follows successful guard removal + directory fsync.
        verify(); _require(not os.path.lexists(budget.pending_path))
        receipt = _encode(dict(version=1, kind='durable-source-retirement', plan_sha256=plan_id,
            source_sha256=plan['source_sha256'], tombstone_sha256=sha(tombstone),
            marker_sha256=sha(marker), retired_identity=_identity(budget)))
        _write_private(budget, directory/'retirement.json', receipt)
        return sha(receipt)


def _receipt(budget, directory, plan_id, plan, files, approval):
    digest(approval)
    raw = _read_private(budget, directory/'retirement.json'); _require(sha(raw) == approval)
    receipt = _decode(raw)
    _fields(receipt, 'version kind plan_sha256 source_sha256 tombstone_sha256 marker_sha256 retired_identity')
    tombstone = _tombstone(plan_id, files)
    marker = _encode(dict(plan_sha256=plan_id, tombstone_sha256=sha(tombstone)))
    _identity_shape(receipt['retired_identity'])
    _require(type(receipt['version']) is int and receipt['version'] == 1
             and receipt['kind'] == 'durable-source-retirement' and receipt['plan_sha256'] == plan_id
             and receipt['source_sha256'] == plan['source_sha256'] and receipt['tombstone_sha256'] == sha(tombstone)
             and receipt['marker_sha256'] == sha(marker)
             and all(receipt['retired_identity'][k] == plan['source_identity'][k]
                     for k in plan['source_identity'] if k != 'ledger')
             and receipt['retired_identity']['ledger']['owner'] == plan['source_identity']['ledger']['owner'])
    return raw


def activate_destination(budget, plan_id, *, approved_plan_sha256, approved_retirement_sha256, writers_stopped):
    """Separate approval attests independently verified source retirement evidence."""
    with _operation(), budget._physical_lock():
        _require(writers_stopped is True and approved_plan_sha256 == plan_id)
        directory, raw, plan, files, state = _packet(budget, plan_id)
        _matches(budget, plan['destination_identity'])
        _receipt(budget, directory, plan_id, plan, files, approved_retirement_sha256)
        _require(not os.path.lexists(directory/'activation.started'))

        def unchanged():
            _matches(budget, plan['destination_identity']); _deadline(budget, plan)
            _require(budget._read_bytes() == files['source.v2.json']
                     and _read_private(budget, budget.directory/WINDOW_NAME) == files['previous-window.json'])
        unchanged()

        def install_window():
            unchanged()
            _write_private(budget, directory/'activation.started', _encode(dict(retirement_sha256=approved_retirement_sha256)))
            next_window = directory/'window.next.json'; _write_private(budget, next_window, raw)
            os.replace(next_window, budget.directory/WINDOW_NAME); _sync(directory); _sync(budget.directory)

        def verify():
            _matches(budget, plan['destination_identity'], ledger=False)
            _require(budget._read_bytes() == files['candidate.v2.json'] and budget._read()[2] == state
                     and _read_private(budget, budget.directory/WINDOW_NAME) == raw)
            _write_private(budget, directory/'activation.completed',
                           _encode(dict(plan_sha256=plan_id, retirement_sha256=approved_retirement_sha256)))
        budget._settle(files['candidate.v2.json'], before_replace=install_window, verify=verify)
        return plan_id


def check_request(budget, plan, request, corpus, ledger):
    """Called under the durable reservation lock, before any resources are opened."""
    from .live_access import require_local_access, require_local_request
    _require(budget.before_request is require_local_access)
    require_local_request(budget.root, backend='pinecone', graph=False)
    plan_id = sha(_encode(plan))
    _require(os.environ.get('NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL') == run_approval(plan_id))
    verified, files = _activated_packet(budget, plan_id)
    _require(verified == plan)
    expected = dict(invocation_id=request.invocation_id, question_sha256=sha(request.question.encode()),
        mode=request.mode, backend=request.backend, profile=request.profile,
        reference_date=request.reference_date.isoformat(), model=MODEL_ID, corpus_sha256=corpus)
    # Equality to the immutable candidate permits zero appended rows only.
    # Even a zero-transport or interrupted reservation burns this one-shot permit.
    _require(expected == plan['request'] and ledger == _decode(files['candidate.v2.json'])
             and budget._read_bytes() == files['candidate.v2.json'])


def _activated_packet(budget, plan_id):
    """Shared read-only validation; caller holds the existing authority lock."""
    directory, _, plan, files, _ = _packet(budget, plan_id)
    _matches(budget, plan['destination_identity'], ledger=False)
    completed = _decode(_read_private(budget, directory/'activation.completed'))
    _fields(completed, 'plan_sha256 retirement_sha256'); _require(completed['plan_sha256'] == plan_id)
    _receipt(budget, directory, plan_id, plan, files, completed['retirement_sha256'])
    return plan, files


def one_shot_selected() -> bool:
    """Presence selects the restricted path; malformed settings never fall back."""
    return any(name in os.environ for name in (
        'NORTHSTAR_AUTHORITY_HANDOFF_PLAN', 'NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL'))


def configured_plan() -> str:
    """Trusted process selection of an existing approval, not new spending authority."""
    from .live_access import local_backend
    plan_id = os.environ.get('NORTHSTAR_AUTHORITY_HANDOFF_PLAN')
    digest(plan_id)
    _require(os.environ.get('NORTHSTAR_AUTHORITY_HANDOFF_RUN_APPROVAL') == run_approval(plan_id)
             and os.environ.get('NORTHSTAR_DEMO_BATCH_ENABLED', '0') == '0'
             and os.environ.get('NORTHSTAR_UI_ENABLE_GRAPH', '0') == '0'
             and local_backend() == 'pinecone')
    return plan_id


@dataclass(frozen=True)
class InvocationStatus:
    """Internal projection of the existing contract; never itself a reservation."""
    plan_sha256: str
    invocation_id: str
    question_sha256: str
    mode: str
    reference_date: date
    available: bool


def inspect_invocation(budget, plan_id: str, *, require_browser: bool = True) -> InvocationStatus:
    """No writes or resource setup. Revalidate on every display/startup/request."""
    from .live_access import require_live_budget, require_local_access, require_local_request
    with _operation():
        _require(plan_id == configured_plan() and budget.before_request is require_local_access)
        require_live_budget(budget.root, budget)
        if require_browser:
            require_local_request(budget.root, backend='pinecone', graph=False)
        with budget._locked(diagnostic_approval=plan_id) as window:
            plan, files = _activated_packet(budget, plan_id); _require(window == plan)
            ledger, _, _ = budget._read()
            baseline = _decode(files['candidate.v2.json'])
            prefix = deepcopy(ledger); prefix['workflows'] = prefix['workflows'][:len(baseline['workflows'])]
            tail = ledger['workflows'][len(baseline['workflows']):]
            _require(prefix == baseline and len(tail) <= 1)
            contract = plan['request']
            if tail:
                row = tail[0]
                _require(row['id'] == contract['invocation_id'] and row['profile'] == LINEAR
                         and row['mode'] == contract['mode'] and row['backend'] == 'pinecone'
                         and row['price_id'] == baseline['active_pricing'])
            claim = contained_path(budget.root, f'.local/linear-demo-results/{contract["invocation_id"]}.json')
            return InvocationStatus(plan_id, contract['invocation_id'], contract['question_sha256'],
                                    contract['mode'], date.fromisoformat(contract['reference_date']),
                                    not tail and not os.path.lexists(claim))


def run_approval(plan_id):
    digest(plan_id)
    return sha(('local-authority-one-shot-run:' + plan_id).encode())


def run_handoff_question(budget, request, *, plan_sha256, approved_run_sha256, consent, opener, limits):
    """Explicit one-shot local adapter; no CLI, UI defaults or hosted access bypass."""
    from .local_live import run_question
    _require(consent is True and approved_run_sha256 == run_approval(plan_sha256)
             and request.profile == LINEAR and request.backend == 'pinecone')
    return run_question(budget, request, opener=opener, limits=limits, diagnostic_approval=plan_sha256)
