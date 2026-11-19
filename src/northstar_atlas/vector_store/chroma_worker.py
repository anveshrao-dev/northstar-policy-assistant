"""Internal isolated Chroma worker. No paid provider or application authorization."""
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tomllib

from northstar_atlas.embeddings import EmbeddingModel
from northstar_atlas.project import contained_path
from .chroma_process import METHODS, receive, send
from .records import IndexConfig


def validate_boundary(root, lock_fd):
    work = Path.cwd()
    if (root != root.resolve(strict=True) or not root.is_absolute()
            or '..' in root.parts or work != work.resolve(strict=True)
            or work.parent != contained_path(root, '.local')
            or not work.name.startswith('chroma-worker-') or any(work.iterdir())):
        raise ValueError('invalid worker boundary')
    st = work.stat()
    if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        raise ValueError('invalid worker permissions')
    index = contained_path(root, '.local/chroma')
    lock = contained_path(root, '.local/chroma/.index.lock')
    actual, inherited = lock.stat(), os.fstat(lock_fd)
    if (not stat.S_ISREG(inherited.st_mode) or inherited.st_uid != os.getuid()
            or inherited.st_nlink != 1
            or (actual.st_dev, actual.st_ino) != (inherited.st_dev, inherited.st_ino)):
        raise ValueError('invalid inherited lock')
    # A separate open description must be unable to acquire the parent's lock.
    with lock.open('rb') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError('index lock is not held')
    if tomllib.loads(contained_path(root, 'pyproject.toml').read_text())['project']['name'] != 'northstar-atlas':
        raise ValueError('invalid project')
    for path in index.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in index')
    return work, index


def install_guards(root, work, index):
    def deny(*args, **kwargs):
        raise OSError('network unavailable in local Chroma worker')
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    socket.getaddrinfo = deny

    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'os.chdir'):
            raise PermissionError('operation forbidden in Chroma worker')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            original = Path(os.fsdecode(args[0])).absolute()
            path = original.resolve()
            private_names = ('.env', 'secrets.toml', 'budget.json')
            if (original.name in private_names or path.name in private_names
                    or (path.is_relative_to(root/'.local') and
                        not path.is_relative_to(work) and not path.is_relative_to(index))):
                raise PermissionError('private configuration unavailable in Chroma worker')
    sys.addaudithook(audit)


def initialize(index, arguments):
    if set(arguments) != {'config', 'create'} or type(arguments['create']) is not bool:
        raise ValueError('invalid initialization')
    config = json.loads(arguments['config'])
    config['model'] = EmbeddingModel(**config['model'])
    if IndexConfig(**config).serialized != arguments['config']:
        raise ValueError('invalid index configuration')
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError
    from .store import COLLECTION_NAME
    client = chromadb.PersistentClient(path=str(index), settings=Settings(
        _env_file=None, chroma_api_impl='chromadb.api.rust.RustBindingsAPI',
        anonymized_telemetry=False, chroma_otel_granularity='none',
        chroma_otel_collection_endpoint='', allow_reset=False))
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        if not arguments['create']:
            return None, 'missing'
        collection = client.create_collection(COLLECTION_NAME, embedding_function=None,
            metadata={'index_config':arguments['config']}, configuration={'hnsw':{'space':'cosine'}})
    configuration = collection.configuration_json
    if ((collection.metadata or {}).get('index_config') != arguments['config']
            or configuration.get('hnsw', {}).get('space') != 'cosine'
            or configuration.get('embedding_function') not in (None, {'type':'legacy'})):
        return None, 'configuration'
    return collection, None


def main(argv=None):
    channel = None
    sequence = 0
    try:
        args = sys.argv[1:] if argv is None else argv
        if len(args) != 3:
            return 1
        channel = socket.socket(fileno=int(args[0]))
        root = Path(args[2])
        work, index = validate_boundary(root, int(args[1]))
        install_guards(root, work, index)
        collection = None
        while sequence < 1024:
            request = receive(channel)
            sequence += 1
            if (not isinstance(request, dict) or set(request) != {'id', 'method', 'arguments'}
                    or type(request['id']) is not int or request['id'] != sequence
                    or not isinstance(request['arguments'], dict)):
                raise ValueError('invalid request')
            method, arguments = request['method'], request['arguments']
            if sequence == 1 and method == 'initialize':
                collection, error = initialize(index, arguments)
                if error:
                    send(channel, {'id':sequence, 'error':error})
                    return 1
                value = {'metadata':collection.metadata, 'configuration':collection.configuration_json}
            elif collection is not None and method in METHODS:
                value = getattr(collection, method)(**arguments)
                if method == 'get' and value.get('embeddings') is not None:
                    value['embeddings'] = value['embeddings'].tolist()
            else:
                raise ValueError('unsupported operation')
            send(channel, {'id':sequence, 'result':value})
        return 1
    except EOFError:
        return 0
    except Exception:
        # External dependency errors are terminal and sanitized, never ignored.
        if channel is not None:
            try:
                send(channel, {'id':sequence, 'error':'worker'})
            except (OSError, ValueError):
                pass
        return 1
    finally:
        if channel is not None:
            channel.close()


if __name__ == '__main__':
    raise SystemExit(main())
