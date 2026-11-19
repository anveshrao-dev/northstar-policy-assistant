"""Bounded local IPC for Chroma; the application never imports the dependency.

A worker lives for one locked index operation. It receives no application
environment or paid authority. The inherited lock stays held if the parent dies.
"""
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

from northstar_atlas.project import contained_path
from .records import VectorStoreError

MAX_FRAME = 16 * 1024 * 1024
TIMEOUT = 60
METHODS = frozenset(('get', 'count', 'query', 'upsert', 'delete', 'add', 'update'))
ERRORS = {
    'missing': 'incomplete index: collection missing; rerun index',
    'configuration': 'incompatible collection model/configuration; do not mix embedding spaces',
    'worker': 'isolated Chroma operation failed; no result accepted',
}


def send(channel, value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(data) > MAX_FRAME:
        raise ValueError('oversized Chroma frame')
    channel.sendall(struct.pack('!I', len(data)) + data)


def receive(channel, *, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Chroma response timed out')
                channel.settimeout(remaining)
            block = channel.recv(size - len(parts))
            if not block:
                raise EOFError
            parts.extend(block)
        return bytes(parts)
    size, = struct.unpack('!I', exact(4))
    if not 0 < size <= MAX_FRAME:
        raise ValueError('invalid Chroma frame size')
    def invalid(_):
        raise ValueError('nonfinite Chroma frame')
    return json.loads(exact(size), parse_constant=invalid)


class ChromaCollection:
    """Collection subset; owner must close it before relinquishing the index lock."""
    def __init__(self, root: Path, config: str, *, create: bool, lock_fd: int):
        self.process = self.channel = self.temporary = None
        self.sequence = 0
        self.closed = False
        try:
            base = contained_path(root, '.local')
            self.temporary = tempfile.TemporaryDirectory(prefix='chroma-worker-', dir=base)
            work = Path(self.temporary.name)
            parent, child = socket.socketpair()  # Anonymous local IPC, no listener/network.
            self.channel = parent
            parent.settimeout(TIMEOUT)
            try:
                # Explicit environment: no credentials, Chroma settings, proxy,
                # HOME, PYTHONPATH, access flags or budget/authority selectors.
                env = dict(PATH='/usr/bin:/bin', TMPDIR=str(work), XDG_CACHE_HOME=str(work),
                           PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
                self.process = subprocess.Popen(
                    [sys.executable, '-I', '-B', '-m', 'northstar_atlas.vector_store.chroma_worker',
                     str(child.fileno()), str(lock_fd), str(root)],
                    cwd=work, env=env, pass_fds=(child.fileno(), lock_fd),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                child.close()
            info = self._request('initialize', {'config': config, 'create': create})
            if not isinstance(info, dict) or set(info) != {'metadata', 'configuration'}:
                raise ValueError('invalid Chroma initialization')
            self.metadata, self.configuration_json = info['metadata'], info['configuration']
        except VectorStoreError:
            self.close()
            raise
        except (OSError, ValueError, EOFError, TypeError):
            self.close()
            raise VectorStoreError(ERRORS['worker']) from None
        except BaseException:
            self.close()
            raise

    def _request(self, method, arguments):
        if self.closed or (method != 'initialize' and method not in METHODS):
            raise VectorStoreError(ERRORS['worker'])
        self.sequence += 1
        try:
            self.channel.settimeout(TIMEOUT)
            send(self.channel, {'id': self.sequence, 'method': method, 'arguments': arguments})
            response = receive(self.channel, timeout=TIMEOUT)
            if (not isinstance(response, dict) or response.get('id') != self.sequence
                    or type(response.get('id')) is not int):
                raise ValueError('invalid Chroma response')
            if set(response) == {'id', 'error'} and response['error'] in ERRORS:
                raise VectorStoreError(ERRORS[response['error']])
            if set(response) != {'id', 'result'}:
                raise ValueError('invalid Chroma response')
            return response['result']
        except VectorStoreError:
            self.close()
            raise
        except (OSError, ValueError, EOFError, TypeError):
            self.close()
            raise VectorStoreError(ERRORS['worker']) from None
        except BaseException:
            self.close()
            raise

    def get(self, **kwargs):
        result = self._request('get', kwargs)
        if result.get('embeddings') is not None:
            # Preserve Chroma's existing ndarray result contract for all callers.
            import numpy as np
            result['embeddings'] = np.asarray(result['embeddings'])
        return result
    def count(self): return self._request('count', {})
    def query(self, **kwargs): return self._request('query', kwargs)
    def upsert(self, **kwargs): return self._request('upsert', kwargs)
    def delete(self, **kwargs): return self._request('delete', kwargs)
    def add(self, **kwargs): return self._request('add', kwargs)
    def update(self, **kwargs): return self._request('update', kwargs)

    def close(self):
        self.closed = True
        if self.channel is not None:
            self.channel.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.temporary is not None:
            self.temporary.cleanup()
