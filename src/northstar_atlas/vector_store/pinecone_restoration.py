"""Explicit no-upsert restoration of an owned, active failed-audit publication.

An immutable local plan is not permission: application needs the reviewed digest
and explicit promotion consent. No old audit report authorizes promotion. Every
application performs fresh HTTPX record and exact-ID visibility verification.
Pending transitions block cooperating readers/writers; this is not an OS fence.
"""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .pinecone_cloud import check_confirmation, load_settings, local_path
from .pinecone_diagnostic import BudgetTransport, DescribeClient, DiagnosticError
from .pinecone_documents import DocumentsIndex, REASONS, _object, validate_egress
from .pinecone_state import PublicationState
from .pinecone_store import PineconeStore, PineconeTarget
from .records import VectorStoreError, canonical_json
from . import pinecone_workflow as workflow
from threading import Lock

LIMIT = 200
GUARD = 'restoration.pending'
STAGES = {'confirmation', 'describe', 'records', 'visibility', 'final_validation',
          'promotion', 'receipt', 'guard_removal', 'complete'}
REASONS_ALLOWED = set(REASONS) | STAGES | {'none', 'budget_exhausted', 'unexpected_operation',
    'target_validation', 'interrupted', 'persistence_failure', 'unexpected_failure'}


class RestorationError(VectorStoreError):
    def __init__(self):
        super().__init__('restoration precondition failed; stop for independent review')


def _require(condition):
    if not condition: raise RestorationError()


def _sha(raw): return hashlib.sha256(raw).hexdigest()
def _encode(value): return (canonical_json(value) + '\n').encode('utf-8')


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_object,
            parse_constant=lambda _: (_ for _ in ()).throw(RestorationError()))
        _require(type(value) is dict)
        return value
    except (ValueError, UnicodeError, TypeError):
        raise RestorationError() from None


def _digest(value):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None)
    return value


def _now(value):
    value = datetime.now(timezone.utc) if value is None else value
    _require(isinstance(value, datetime) and value.utcoffset() is not None)
    return value.astimezone(timezone.utc)


def _identity(path):
    info = path.stat()
    _require(not path.is_symlink() and info.st_uid == os.getuid())
    return [info.st_dev, info.st_ino, info.st_uid]


def _sync(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _create(path, raw):
    """Exclusive durable creation; partial artifacts stay in place on failure."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    _sync(path.parent)
    _require(path.read_bytes() == raw)


def _clear_guard(path, raw):
    _require(path.read_bytes() == raw)
    try:
        path.unlink()
        _sync(path.parent)
    except BaseException:
        # A caught cleanup error must not expose a possibly committed transition.
        # If the filesystem itself is unavailable, recovery still requires review.
        if not path.exists(): _create(path, raw)
        raise


def _state(root, publication_id):
    _digest(publication_id)
    root = Path(root).resolve(strict=True)
    directory = local_path(root, '.local/pinecone/' + publication_id)
    _require(directory.is_dir())
    pointer = _json((directory / 'active.json').read_bytes())
    _require(set(pointer) == {'namespace'})
    namespace = pointer['namespace']; PublicationState.validate_namespace(namespace)
    raw = local_path(root, str((directory / (namespace + '.json')).relative_to(root))).read_bytes()
    run = _json(raw)
    try: target = PineconeTarget(**run['target'])
    except (KeyError, TypeError, ValueError): raise RestorationError() from None
    state = PublicationState(root, asdict(target))
    _require(state.path == directory)
    state.check_paths()
    _identity(root); _identity(directory)
    return state, target


def _context(state, target):
    state.check_paths()
    namespace = state.active()
    _require(namespace is not None)
    snapshot = workflow.load_snapshot(state.root, namespace)
    workflow.validate_m33(snapshot, state.root)
    store = PineconeStore(state.root, snapshot, target, documents=None, describe_target=None,
        current_fingerprint=lambda: workflow.corpus_snapshot(state.root)[1], poll_attempts=1)
    run = state.read_run(namespace)
    _require(type(run) is dict and run.get('state') == 'incomplete' and run.get('phase') == 'failed audit')
    store._owned(run); store._source_check()
    paths = {'manifest': state.path / (namespace + '.json'), 'pointer': state.path / 'active.json',
             'snapshot': workflow._snapshot_path(state.root, namespace),
             'lock': state.path / 'writer.lock',
             'target': local_path(state.root, '.local/pinecone/target.json'),
             'confirmation': local_path(state.root, '.local/pinecone/target-confirmation.json')}
    bindings = {k: {'sha256': _sha(p.read_bytes()), 'identity': _identity(p)} for k, p in paths.items()}
    metadata = _json(paths['target'].read_bytes())
    _require(all(metadata.get(k) == v for k, v in asdict(target).items() if k != 'host'))
    candidate = {**run, 'state': 'ready', 'phase': 'restored without upsert', 'verified_records': 189}
    _require(len(store.by_id) == 189)
    return store, paths, bindings, paths['manifest'].read_bytes(), _encode(candidate)


def _plan_path(state, digest): return state.path / ('restoration-' + _digest(digest) + '.json')
def _receipt_path(state, digest): return state.path / ('restoration-' + _digest(digest) + '.receipt.json')


def prepare_restoration(root, publication_id, *, approve_preparation=False, now=None):
    """Local review artifact only; no credential loading or remote operations."""
    _require(approve_preparation is True)
    state, target = _state(root, publication_id)
    with state.lock():
        store, _, bindings, original, candidate = _context(state, target)
        confirmed = _now(now)
        plan = {'version': 1, 'kind': 'no-upsert-restoration', 'publication_id': publication_id,
            'root_identity': _identity(state.root), 'directory_identity': _identity(state.path),
            'target': asdict(target), 'namespace': store.namespace, 'bindings': bindings,
            'original_manifest': original.decode('utf-8'), 'candidate_manifest': candidate.decode('utf-8'),
            'max_http_attempts': LIMIT, 'created_at': confirmed.isoformat(),
            'expires_at': (confirmed + timedelta(hours=24)).isoformat()}
        raw = _encode(plan); digest = _sha(raw)
        _create(_plan_path(state, digest), raw)
        return {'plan_sha256': digest, 'candidate_sha256': _sha(candidate),
                'expires_at': plan['expires_at'], 'max_http_attempts': LIMIT}


def _load_plan(state, target, digest, *, now=None, check_expiry=True):
    path = _plan_path(state, digest)
    try: raw = path.read_bytes()
    except OSError: raise RestorationError() from None
    _require(_sha(raw) == digest)
    plan = _json(raw)
    _require(raw == _encode(plan) and set(plan) == {'version', 'kind', 'publication_id',
        'root_identity', 'directory_identity', 'target', 'namespace', 'bindings',
        'original_manifest', 'candidate_manifest', 'max_http_attempts', 'created_at', 'expires_at'})
    _require(type(plan['version']) is int and plan['version'] == 1
        and plan['kind'] == 'no-upsert-restoration' and plan['publication_id'] == state.path.name
        and plan['root_identity'] == _identity(state.root) and plan['directory_identity'] == _identity(state.path)
        and plan['target'] == asdict(target) and type(plan['max_http_attempts']) is int
        and plan['max_http_attempts'] == LIMIT)
    start, end = (datetime.fromisoformat(plan[k]) for k in ('created_at', 'expires_at'))
    _require(start.utcoffset() is not None and end.utcoffset() is not None and end - start == timedelta(hours=24))
    if check_expiry: _require(start <= _now(now) < end)
    _require(not _receipt_path(state, digest).exists())
    return plan


def _check_binding(state, target, plan):
    _require(plan['root_identity'] == _identity(state.root)
             and plan['directory_identity'] == _identity(state.path))
    store, paths, bindings, original, candidate = _context(state, target)
    _require(bindings == plan['bindings'] and store.namespace == plan['namespace']
        and original.decode('utf-8') == plan['original_manifest']
        and candidate.decode('utf-8') == plan['candidate_manifest'])
    return store, paths, candidate


def _check_promoted(state, store, paths, plan, candidate, candidate_identity, now):
    """Final fence check after persistence, without treating the old manifest as current."""
    state.check_paths()
    _require(plan['root_identity'] == _identity(state.root)
             and plan['directory_identity'] == _identity(state.path))
    _require(paths['manifest'].read_bytes() == candidate
             and _identity(paths['manifest']) == candidate_identity)
    store._owned(_json(candidate))
    for key, path in paths.items():
        if key != 'manifest':
            _require({'sha256': _sha(path.read_bytes()), 'identity': _identity(path)} == plan['bindings'][key])
    store._source_check()
    _require(_now(now) < datetime.fromisoformat(plan['expires_at']))


class RestorationBudget:
    """Shared physical dispatch cap; refused calls consume no transport attempt."""
    def __init__(self):
        self._lock = Lock()
        self._counts = dict.fromkeys(('describe', 'list', 'fetch', 'search'), 0)
        self.last_operation = None

    def reserve(self, operation):
        with self._lock:
            if operation not in self._counts: raise DiagnosticError('unexpected_operation')
            self.last_operation = operation
            if sum(self._counts.values()) >= LIMIT: raise DiagnosticError('budget_exhausted')
            self._counts[operation] += 1

    def counts(self):
        with self._lock: return dict(self._counts)


def _report(budget, meter, stage, outcome, reason, verified, visible):
    _require(stage in STAGES and outcome in ('success', 'failed', 'inconclusive') and reason in REASONS_ALLOWED)
    counts = budget.counts(); total = sum(counts.values()); data_total = total - counts['describe']
    _require(total <= LIMIT and 0 <= visible <= verified <= 189)
    validate_egress(meter.egress_bytes if meter.egress_responses else None,
                   meter.egress_responses, meter.documents_usage_responses)
    return {'schema_version': 1, 'outcome': outcome, 'stage': stage, 'reason': reason,
        'attempts': counts, 'max_http_attempts': LIMIT, 'operation': budget.last_operation,
        'verified_records': verified, 'visible_records': visible,
        'validated_read_units': meter.read_units if meter.documents_usage_responses else None,
        'responses_with_validated_usage': meter.documents_usage_responses,
        'usage_complete': data_total > 0 and meter.documents_usage_responses == data_total,
        'validated_egress_bytes': meter.egress_bytes if meter.egress_responses else None,
        'responses_with_validated_egress': meter.egress_responses,
        'egress_complete': data_total > 0 and meter.egress_responses == data_total}


def apply_restoration(root, publication_id, *, approved_plan_sha256, approve_promotion=False,
                      settings=None, control_transport=None, documents_transport=None, now=None):
    """One separately approved execution; no automatic retry, rollback or repair."""
    _require(approve_promotion is True); _digest(approved_plan_sha256)
    state, target = _state(root, publication_id)
    with state.lock():
        plan = _load_plan(state, target, approved_plan_sha256, now=now)
        store, paths, candidate = _check_binding(state, target, plan)
        guard = state.path / GUARD
        guard_raw = _encode({'version': 1, 'plan_sha256': approved_plan_sha256})
        # Exclusive/fsynced guard before credential load or any remote request.
        _create(guard, guard_raw)
        budget = RestorationBudget(); meter = workflow.RequestMeter()
        stage, verified, visible = 'confirmation', 0, 0
        try:
            if settings is None: settings = load_settings(state.root)
            check_confirmation(state.root, settings)
            _require(all(getattr(settings, k) == v for k, v in asdict(target).items() if k != 'host'))
            with ExitStack() as stack:
                control = stack.enter_context(DescribeClient(settings, budget, transport=control_transport))
                allowed = {('POST', f'{target.host}/namespaces/{store.namespace}/documents/{op}'): op
                           for op in ('list', 'fetch', 'search')}
                transport = BudgetTransport(budget, allowed, documents_transport)
                index = stack.enter_context(DocumentsIndex(host=target.host, api_key=settings.api_key,
                    transport=transport, observe=meter.observe_documents, observe_egress=meter.observe_egress,
                    on_failure=meter.record_documents_failure))
                store.documents = index.documents; store.describe_target = control.describe
                stage = 'describe'; store._target_check()
                deadline = time.monotonic() + 300
                stage = 'records'; store._audit(deadline); verified = 189
                stage = 'visibility'
                # Single pass, one exact-record probe each, no eventual-consistency retry.
                for key in sorted(store.by_id):
                    _require(time.monotonic() < deadline)
                    found = store._dense_search(store.by_id[key].vector, 1, {'record_id': {'$eq': key}})
                    _require(len(found) == 1 and store._decode(found[0].to_dict(), with_vector=False).chunk_id == key)
                    visible += 1
                stage = 'final_validation'
                _check_binding(state, target, plan)
                check_confirmation(state.root, settings)
                store._target_check(); store._source_check()
                _require(_now(now) < datetime.fromisoformat(plan['expires_at']))
            # No active-pointer publish call. Recheck bytes/identities after clients close.
            _check_binding(state, target, plan)
            _require(guard.read_bytes() == guard_raw)
            _require(_now(now) < datetime.fromisoformat(plan['expires_at']))
            stage = 'promotion'
            state.write_run(store.namespace, _json(candidate))
            candidate_identity = _identity(paths['manifest'])
            _check_promoted(state, store, paths, plan, candidate, candidate_identity, now)
            stage = 'receipt'
            report = _report(budget, meter, 'complete', 'success', 'none', verified, visible)
            _create(_receipt_path(state, approved_plan_sha256), _encode({
                'plan_sha256': approved_plan_sha256, 'candidate_sha256': _sha(candidate),
                'outcome': 'promoted', 'report': report}))
            workflow._save_private_report(state.root, 'restoration', report)
            _check_promoted(state, store, paths, plan, candidate, candidate_identity, now)
            stage = 'guard_removal'; _clear_guard(guard, guard_raw)
            return report
        except BaseException as exc:
            # Never undo the manifest or clear a guard after an interrupted operation.
            reason = getattr(exc, 'reason', None)
            reason = reason if reason in REASONS_ALLOWED else (
                'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else
                'persistence_failure' if isinstance(exc, OSError) else
                stage if isinstance(exc, (ValueError, VectorStoreError)) else 'unexpected_failure')
            outcome = 'inconclusive' if reason in ('interrupted', 'persistence_failure', 'budget_exhausted', 'unexpected_failure') else 'failed'
            report = _report(budget, meter, stage, outcome, reason, verified, visible)
            workflow._save_private_report(state.root, 'restoration', report)
            return report


def abort_restoration(root, publication_id, *, approved_plan_sha256, approved_guard_sha256, approve_abort=False):
    """Separately approved local abort ONLY before manifest promotion.

    Post-replace ambiguity is blocked; this operation cannot roll back readiness,
    trust a prior report, replay remote verification, or reactivate an approval.
    """
    _require(approve_abort is True); _digest(approved_guard_sha256)
    state, target = _state(root, publication_id)
    with state.lock(allow_restoration_pending=True):
        plan = _load_plan(state, target, approved_plan_sha256, check_expiry=False)
        _check_binding(state, target, plan)
        guard = state.path / GUARD; raw = guard.read_bytes()
        _require(_sha(raw) == approved_guard_sha256
                 and raw == _encode({'version': 1, 'plan_sha256': approved_plan_sha256}))
        _create(_receipt_path(state, approved_plan_sha256), _encode({
            'plan_sha256': approved_plan_sha256, 'outcome': 'aborted_before_promotion'}))
        _clear_guard(guard, raw)
        return {'outcome': 'aborted_before_promotion', 'state': 'incomplete'}
