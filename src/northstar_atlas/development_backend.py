"""Local publication bindings for an independently approved development capability.

No credentials, transport, publication creation or readiness promotion. A binding
is evidence to review, not authorization; the development approval chain owns it.
"""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import os
import stat

from .budget_transition import _decode, _sha, _digest
from .live_budget import _require, _timestamp
from .project import contained_path


@contextmanager
def _publication_lock(state):
    # Unlike publication preparation, this must never create a missing lock/root.
    state.check_paths()
    path = state.path / 'writer.lock'
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                 and info.st_nlink == 1 and (info.st_dev, info.st_ino) ==
                 (path.lstat().st_dev, path.lstat().st_ino))
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.require_no_transition()
        yield
    finally:
        os.close(fd)


def _file(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                 and info.st_nlink == 1 and not stat.S_IMODE(info.st_mode) & 0o022
                 and (info.st_dev, info.st_ino) == (path.lstat().st_dev, path.lstat().st_ino))
        raw = stream.read(16_000_001)
    _require(len(raw) <= 16_000_000)
    return raw, dict(sha256=_sha(raw), identity=[info.st_dev, info.st_ino])


def capture_publication(root, publication_id):
    """Check the existing ready, owned 189-record snapshot using local bytes only."""
    from .vector_store.pinecone_restoration import _state
    from .vector_store import pinecone_workflow as w
    from .vector_store.pinecone_store import PineconeStore
    _digest(publication_id)
    state, target = _state(root, publication_id)
    with _publication_lock(state):
        namespace = state.active()
        snapshot = w.load_snapshot(root, namespace)
        w.validate_m33(snapshot, root)
        store = PineconeStore(root, snapshot, target, documents=None, describe_target=None,
            current_fingerprint=lambda: w.corpus_snapshot(root)[1])
        run = state.read_run(namespace)
        _require(run is not None and run.get('state') == 'ready' and len(store.by_id) == 189)
        store._owned(run); store._source_check()
        paths = dict(manifest=state.path/(namespace+'.json'), pointer=state.path/'active.json',
            snapshot=w._snapshot_path(root, namespace), lock=state.path/'writer.lock',
            target=contained_path(root,'.local/pinecone/target.json'),
            confirmation=contained_path(root,'.local/pinecone/target-confirmation.json'))
        files = {k:_file(p) for k,p in paths.items()}
        metadata = _decode(files['target'][0]); confirmation = _decode(files['confirmation'][0])
        _require(set(metadata) == {'name','cloud','region','organization','project_name','project_id','plan'}
                 and all(metadata.get(k)==v for k,v in asdict(target).items() if k!='host'))
        for key in ('organization','project_name','project_id','cloud','region','plan'):
            _require(confirmation.get(key)==metadata[key])
        _require(confirmation.get('index_name')==target.name)
        _digest(confirmation.get('key_digest')); _timestamp(confirmation.get('confirmed_at'))
        return dict(publication_id=publication_id, target=asdict(target), namespace=namespace,
            snapshot_id=snapshot.snapshot_id, corpus_fingerprint=snapshot.corpus_fingerprint,
            files={k:v[1] for k,v in files.items()})


def validate_binding(value):
    """Validate historical approval shape without requiring its publication online."""
    from .vector_store.pinecone_store import PineconeTarget
    from .vector_store.pinecone_state import PublicationState
    from .vector_store.records import digest
    _require(type(value) is dict and set(value)=={'publication_id','target','namespace','snapshot_id',
                                                 'corpus_fingerprint','files','review_sha256'})
    for key in ('publication_id','snapshot_id','corpus_fingerprint','review_sha256'): _digest(value[key])
    target=PineconeTarget(**value['target'])
    _require(value['target']==asdict(target) and digest(asdict(target))==value['publication_id'])
    PublicationState.validate_namespace(value['namespace'])
    _require(value['namespace']=='dev-'+value['snapshot_id'])
    _require(type(value['files']) is dict and set(value['files'])==
             {'manifest','pointer','snapshot','lock','target','confirmation'})
    for entry in value['files'].values():
        _require(type(entry) is dict and set(entry)=={'sha256','identity'})
        _digest(entry['sha256'])
        identity=entry['identity']
        _require(type(identity) is list and len(identity)==2 and all(type(n) is int and n>=0 for n in identity))


def check_publication(root, binding):
    validate_binding(binding)
    current = capture_publication(root, binding['publication_id'])
    _require(current == {k:v for k,v in binding.items() if k!='review_sha256'})
