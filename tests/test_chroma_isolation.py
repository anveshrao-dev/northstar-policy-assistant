"""Real isolated Chroma processes with synthetic configuration and fake vectors."""
import os
import fcntl
import json
from pathlib import Path
import socket
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from checkout_fixture import ROOT, checkout


PROBE = r'''
import json, os, socket, sys
from datetime import date
from pathlib import Path
root = Path.cwd()
before = dict(os.environ)
def deny(*args, **kwargs):
    raise OSError('offline fixture')
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
socket.getaddrinfo = deny
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = Path(os.fsdecode(args[0]))
        if p.name in ('.env', 'secrets.toml', 'budget.json'):
            raise PermissionError('private configuration access forbidden')
sys.addaudithook(guard)
from northstar_atlas.embeddings import EmbeddingModel, embed_chunks
from northstar_atlas.ingestion.document_loader import load_documents
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.vector_store import ChromaStore, IndexConfig
from northstar_atlas.vector_store.preparation import corpus_snapshot
class Vectors:
    model = EmbeddingModel('synthetic-only', 'revision-1', 3, 'l2')
    max_input_tokens = 256
    def token_counts(self, texts): return [len(t.split())+2 for t in texts]
    def embed_documents(self, texts): return [[1., 0., 0.] for t in texts]
    def embed_query(self, text): return [1., 0., 0.]
provider = Vectors()
docs = load_documents(root, mode='draft_review')
chunks = [chunk_document(d).chunks[0] for d in docs[:2]]
store = ChromaStore(root, IndexConfig(provider.model, 256))
_, fingerprint = corpus_snapshot(root)
assert store.synchronize(embed_chunks(chunks, provider), corpus_fingerprint=fingerprint).total == 2
assert store.status()['state'] == 'ready'
args = dict(corpus_fingerprint=fingerprint, reference_date=date(2026,10,22))
approved = store.search('policy', provider, **args)
draft = store.search('policy', provider, mode='draft_review', **args)
assert len(approved) == 1 and len(draft) == 2
assert {h.chunk for h in draft} == set(chunks)
assert 'chromadb' not in sys.modules, 'Chroma imported in application process'
assert Path.cwd() == root and dict(os.environ) == before
assert not list((root/'.local').glob('chroma-worker-*')), 'worker directory leaked'
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
with patch('northstar_atlas.local_embeddings.MiniLMProvider', return_value=provider), \
     patch('northstar_atlas.nebius.NebiusProvider', side_effect=AssertionError('no paid inference')), \
     patch('northstar_atlas.live_budget.configured_budget', side_effect=AssertionError('no accounting')):
    app = AppTest.from_file(str(root/'src/northstar_atlas/ui/app.py'), default_timeout=30).run()
    assert not app.exception and app.radio(key='experience').value == 'Offline demo'
    app.radio(key='experience').set_value('Policy search — retrieval only').run()
    app.chat_input[0].set_value('Who approves access?').run()
    assert not app.exception
    view = app.session_state['conversation'].history[-1].view
    assert view.kind == 'evidence' and view.passages
    app.radio(key='experience').set_value('Live Q&A').run()
    assert app.chat_input[0].disabled
assert 'chromadb' not in sys.modules
assert not list((root/'.local').glob('chroma-worker-*'))
print(json.dumps({'approved':len(approved), 'draft':len(draft), 'isolated':True}))
'''


class ChromaIsolationTests(unittest.TestCase):
    def test_real_retrieval_never_reads_cwd_dotenv_or_imports_chroma_in_parent(self):
        for content in (None, 'CHROMA_SERVER_HTTP_PORT=43817\nALLOW_RESET=true\n',
                        'CHROMA_SERVER_HTTP_PORT=not-an-integer\n'):
            with self.subTest(dotenv='absent' if content is None else 'synthetic'):
                root = checkout(self)
                (root/'.local/tmp').mkdir(parents=True)
                if content is not None:
                    (root/'.env').write_text(content)
                probe = root/'probe.py'
                probe.write_text(PROBE)
                env = {'PATH':str(ROOT/'.venv/bin')+':/usr/bin:/bin',
                       'TMPDIR':str(root/'.local/tmp'), 'PYTHONDONTWRITEBYTECODE':'1',
                       'NORTHSTAR_PROJECT_ROOT':str(root), 'HF_HUB_OFFLINE':'1',
                       'TRANSFORMERS_OFFLINE':'1', 'NEBIUS_API_KEY':'synthetic-do-not-inherit',
                       'CHROMA_SERVER_HTTP_PORT':'invalid-process-setting'}
                result = subprocess.run([sys.executable, '-I', '-B', str(probe)], cwd=root,
                    env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"isolated": true', result.stdout)
                if content is not None:
                    self.assertEqual((root/'.env').read_text(), content)


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        from northstar_atlas.vector_store import ChromaStore, IndexConfig
        from test_vector_store import FakeProvider
        self.root = checkout(self)
        self.provider = FakeProvider()
        self.store = ChromaStore(self.root, IndexConfig(self.provider.model, 256))

    def seed(self):
        from northstar_atlas.embeddings import embed_chunks
        from northstar_atlas.ingestion.document_loader import load_documents
        from northstar_atlas.ingestion.chunker import chunk_document
        chunks = [chunk_document(load_documents(self.root)[0]).chunks[0]]
        records = embed_chunks(chunks, self.provider)
        self.store.synchronize(records, corpus_fingerprint='synthetic')
        return records

    def test_child_gets_no_inherited_credentials_settings_or_authority(self):
        from northstar_atlas.vector_store import chroma_process
        spawn = subprocess.Popen
        captured = []
        def real_spawn(*args, **kwargs):
            process = spawn(*args, **kwargs)
            captured.append((kwargs, process))
            return process
        with patch.dict(os.environ, NEBIUS_API_KEY='synthetic-secret',
                        CHROMA_SERVER_HTTP_PORT='bad', NORTHSTAR_CONTROLLED_LIVE='1'), \
                patch.object(chroma_process.subprocess, 'Popen', side_effect=real_spawn):
            self.seed()
        options, process = captured[0]
        self.assertFalse(any(k.startswith(('NEBIUS_', 'PINECONE_', 'NORTHSTAR_', 'CHROMA_'))
                             for k in options['env']))
        self.assertNotIn('synthetic-secret', str(options))
        self.assertIsNotNone(process.poll())
        self.assertFalse(Path(options['cwd']).exists())

    def test_crashed_worker_is_not_retried_and_lock_outlives_worker(self):
        from northstar_atlas.vector_store import VectorStoreError, ChromaStore
        self.seed()
        with self.store._lock():
            collection = self.store._collection()
            pid = collection.process.pid
            collection.process.kill()
            collection.process.wait(timeout=10)
            with self.assertRaises(VectorStoreError):
                collection.count()
            self.assertEqual(collection.process.pid, pid)
            with self.assertRaisesRegex(VectorStoreError, 'busy'):
                ChromaStore(self.root, self.store.config).status()
        self.assertFalse(list((self.root/'.local').glob('chroma-worker-*')))
        self.assertEqual(self.store.status()['state'], 'ready')

    def test_interrupted_write_stays_incomplete_and_cleans_up(self):
        from northstar_atlas.vector_store.chroma_process import ChromaCollection
        records = self.seed()
        upsert = ChromaCollection.upsert
        def interrupted(collection, **kwargs):
            upsert(collection, **kwargs)
            raise KeyboardInterrupt
        with patch.object(ChromaCollection, 'upsert', interrupted), self.assertRaises(KeyboardInterrupt):
            self.store.synchronize(records, corpus_fingerprint='changed')
        self.assertEqual(self.store.status()['state'], 'incomplete')
        self.assertFalse(list((self.root/'.local').glob('chroma-worker-*')))

    def test_collection_requires_lock_and_rejects_configuration_mismatch(self):
        from dataclasses import replace
        from northstar_atlas.vector_store import ChromaStore, VectorStoreError
        self.seed()
        with self.assertRaisesRegex(VectorStoreError, 'lock'):
            self.store._collection()
        other = ChromaStore(self.root, replace(self.store.config, token_limit=128))
        with other._lock(), self.assertRaisesRegex(VectorStoreError, 'configuration'):
            other._collection()
        self.assertEqual(self.store.status()['state'], 'ready')

    def test_malformed_protocol_is_terminal_without_raw_error_details(self):
        from northstar_atlas.vector_store import VectorStoreError
        from northstar_atlas.vector_store.chroma_process import send
        self.seed()
        with self.store._lock():
            collection = self.store._collection()
            send(collection.channel, {'id':777, 'method':'synthetic-secret', 'arguments':{}})
            with self.assertRaises(VectorStoreError) as caught:
                collection.count()
            self.assertNotIn('synthetic-secret', str(caught.exception))
            self.assertIsNotNone(collection.process.poll())

    def test_missing_collection_and_failed_spawn_are_safe_and_leave_no_worker(self):
        from northstar_atlas.vector_store import VectorStoreError
        with self.store._lock(), self.assertRaisesRegex(VectorStoreError, 'missing'):
            self.store._collection()
        with patch('northstar_atlas.vector_store.chroma_process.subprocess.Popen', side_effect=OSError('synthetic-secret')):
            with self.store._lock(), self.assertRaises(VectorStoreError) as caught:
                self.store._collection(create=True)
            self.assertNotIn('synthetic-secret', str(caught.exception))
        self.assertFalse(list((self.root/'.local').glob('chroma-worker-*')))

    def test_frame_size_and_timeout_are_bounded(self):
        from northstar_atlas.vector_store.chroma_process import MAX_FRAME, receive
        left, right = socket.socketpair()
        self.addCleanup(left.close); self.addCleanup(right.close)
        right.sendall(struct.pack('!I', MAX_FRAME+1))
        with self.assertRaises(ValueError):
            receive(left, timeout=.1)
        with self.assertRaises(TimeoutError):
            receive(left, timeout=.01)

    def test_boundary_rejects_unlocked_or_wrong_lock_and_nonempty_or_public_cwd(self):
        from northstar_atlas.vector_store.chroma_worker import validate_boundary
        self.store.path.mkdir(parents=True)
        with (self.store.path/'.index.lock').open('a') as lock, \
                tempfile.TemporaryDirectory(dir=self.root/'.local', prefix='chroma-worker-') as name:
            work = Path(name)
            # Only replace cwd observation; never change the running process cwd.
            with patch.object(Path, 'cwd', return_value=work):
                with self.assertRaises(ValueError): validate_boundary(self.root, lock.fileno())
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(validate_boundary(self.root, lock.fileno())[0], work)
                with (self.root/'wrong-lock').open('w') as wrong:
                    with self.assertRaises(ValueError): validate_boundary(self.root, wrong.fileno())
                work.chmod(0o755)
                with self.assertRaises(ValueError): validate_boundary(self.root, lock.fileno())
                work.chmod(0o700)
                (work/'.env').write_text('CHROMA_SERVER_HTTP_PORT=1')
                with self.assertRaises(ValueError): validate_boundary(self.root, lock.fileno())

    def test_child_guards_block_dotenv_alias_private_files_network_and_cwd_change(self):
        work = self.root/'.local/chroma-worker-guard'
        work.mkdir(parents=True, mode=0o700)
        index = self.root/'.local/chroma'; index.mkdir()
        (self.root/'synthetic-value').write_text('not-a-credential')
        (work/'.env').symlink_to(self.root/'synthetic-value')
        private = self.root/'.local/controlled-live'; private.mkdir()
        (private/'budget.json').write_text('{}')
        script = self.root/'guard.py'
        script.write_text('''
import os, socket, sys
from pathlib import Path
from northstar_atlas.vector_store.chroma_worker import install_guards
root = Path(sys.argv[1]); work = Path.cwd()
install_guards(root, work, root/'.local/chroma')
actions = [lambda: (work/'.env').read_text(),
           lambda: (root/'.local/controlled-live/budget.json').read_text(),
           lambda: socket.create_connection(('never.example',443)),
           lambda: os.chdir(root)]
for action in actions:
    try: action()
    except OSError: pass
    else: raise AssertionError('guard bypass')
assert Path.cwd() == work
''')
        result = subprocess.run([sys.executable, '-I', '-B', str(script), str(self.root)],
            cwd=work, env={'PATH':'/usr/bin:/bin'}, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parent_death_keeps_lock_until_worker_exits(self):
        # Stop only this test's child, then abruptly exit its parent. The inherited
        # open-description lock must prevent another writer until child cleanup.
        script = self.root/'crash.py'
        script.write_text('''
import json, os, signal
from pathlib import Path
from northstar_atlas.vector_store import ChromaStore, IndexConfig
from northstar_atlas.embeddings import EmbeddingModel
store = ChromaStore(Path.cwd(), IndexConfig(EmbeddingModel('synthetic','r1',3,'l2'),256))
with store._lock():
    worker = store._collection(create=True)
    os.kill(worker.process.pid, signal.SIGSTOP)
    print(json.dumps({'pid':worker.process.pid}), flush=True)
    os._exit(0)
''')
        run = subprocess.run([sys.executable, '-I', '-B', str(script)], cwd=self.root,
            env={'PATH':'/usr/bin:/bin'}, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        pid = json.loads(run.stdout)['pid']
        try:
            with (self.root/'.local/chroma/.index.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.kill(pid, signal.SIGCONT)
                deadline = time.monotonic()+15
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline: raise
                        time.sleep(.05)
        finally:
            # Never leave a stopped synthetic process if an assertion fails.
            try: os.kill(pid, signal.SIGCONT)
            except ProcessLookupError: pass
