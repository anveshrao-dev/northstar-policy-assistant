"""Contained temporary checkout data; never copies credentials or runtime stores."""
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def checkout(test):
    temporary = tempfile.TemporaryDirectory(dir=ROOT / '.local/tmp')
    test.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    for relative in ('pyproject.toml', 'src/northstar_atlas/__init__.py',
                     'src/northstar_atlas/ui/app.py', 'data/corpus_manifest.json'):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copytree(ROOT / 'data/policies', root / 'data/policies')
    return root


def synthetic_graph_snapshot(root):
    """Freeze public policy spans with authored vectors, not MiniLM output.

    Split 38 canonical chunks into two source-complete pieces to exercise the
    189-record contract. This is structural fixture data, not a retrieval score
    or a reproduction of the historical tokenizer/embedding output. Callers bind
    only the expected snapshot ID/hash in their test scope; validators stay real.
    """
    from dataclasses import asdict, replace
    import hashlib
    import json
    from test_context import family
    from northstar_atlas.embeddings import EmbeddedChunk, EmbeddingModel
    from northstar_atlas.ingestion.chunker import chunk_document
    from northstar_atlas.local_embeddings import MODEL_ID, MODEL_REVISION
    from northstar_atlas.vector_store.catalog import ChunkCatalog
    from northstar_atlas.vector_store.preparation import corpus_snapshot, freeze_snapshot
    from northstar_atlas.vector_store.records import IndexConfig, metadata_for

    root = Path(root).resolve(strict=True)
    if not root.is_relative_to((ROOT / '.local/tmp').resolve(strict=True)):
        raise ValueError('synthetic snapshots require an ephemeral test checkout')
    documents, fingerprint = corpus_snapshot(root)
    canonical = tuple(c for d in documents for c in chunk_document(d).chunks)
    assert len(canonical) == 151
    model = EmbeddingModel(MODEL_ID, MODEL_REVISION, 384, 'l2')
    config = IndexConfig(model, 256)
    pieces = []
    for index, source in enumerate(canonical):
        if index < 38:
            pieces.extend(replace(piece, embedding_model=model)
                          for piece in family(source, [len(source.text) // 2]))
        else:
            pieces.append(source)
    # Real reconstruction/provenance checks reject missing spans or altered text.
    catalog = ChunkCatalog(canonical, pieces, config, fingerprint, root=root)
    assert len(catalog.chunks) == 189
    records = [EmbeddedChunk(piece, tuple(float(i == index % 384) for i in range(384)), model)
               for index, piece in enumerate(pieces)]
    snapshot = freeze_snapshot(records, config, fingerprint)
    data = dict(snapshot_id=snapshot.snapshot_id, corpus_fingerprint=fingerprint,
                config=asdict(config), records=[
                    dict(id=r.chunk_id, text=r.chunk.text,
                         metadata=metadata_for(r.chunk, r.model), vector=list(r.vector))
                    for r in snapshot.records])
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    path = root / '.local/pinecone/snapshots' / f'{snapshot.snapshot_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(raw)
    return path, dict(SNAPSHOT_ID=snapshot.snapshot_id,
                      SNAPSHOT_SHA256=hashlib.sha256(raw).hexdigest())
