"""Build a complete ticker slice from cached filings; never fetch or re-embed."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.financial_rag.corpus_snapshot import build_corpus_snapshot, write_corpus_snapshot
from src.financial_rag.retrieval.local_dense import _load_chunks
from src.financial_rag.retrieval.packed_vectors import load_packed_vectors, pack_vector_cache
from src.financial_rag.storage import LocalRagStore


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def build_slice(*, source: Path, output: Path, tickers: list[str], packed_source: Path | None = None) -> dict:
    wanted = set(tickers)
    if not wanted or len(wanted) != len(tickers):
        raise ValueError('Specify nonempty, unique tickers.')
    if output.exists():
        raise ValueError('Output must be a fresh directory; existing artifacts are never overwritten.')
    store = LocalRagStore(root=output)
    seen: set[str] = set()
    source_tickers: set[str] = set()
    # Retain original file order and rows, including all forms and missing vectors.
    for path in sorted((source / 'data/filings/chunks').glob('*.jsonl')):
        if path.name == 'manifest.jsonl':
            continue
        selected = []
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                ticker = str(row.get('ticker') or row.get('metadata', {}).get('ticker', '')).upper()
                if ticker:
                    source_tickers.add(ticker)
                if ticker in wanted:
                    selected.append(line if line.endswith('\n') else line + '\n')
                    seen.add(ticker)
        if selected:
            (store.chunks_dir / path.name).write_text(''.join(selected), encoding='utf-8')
    if seen != wanted:
        raise ValueError(f'Tickers absent from corpus: {sorted(wanted - seen)}; partial output is not deployable.')
    chunks = _load_chunks(store.chunks_dir)
    ids = {chunk.chunk_id for chunk in chunks}
    if len(ids) != len(chunks):
        raise ValueError('Duplicate selected chunk IDs.')
    if packed_source is None:
        pack_vector_cache(vector_cache_dir=source / 'data/vector_cache',
                          out_dir=store.packed_dir, chunk_ids=ids)
    else:
        packed = load_packed_vectors(packed_source)
        selected_ids = [chunk_id for chunk_id in packed if chunk_id in ids]
        rows = [packed.row_for(chunk_id) for chunk_id in selected_ids]
        matrix = np.asarray(packed.matrix[rows], dtype=np.float16)
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite selected vectors.')
        np.save(store.packed_dir / 'vectors.npy', matrix, allow_pickle=False)
        (store.packed_dir / 'chunk_ids.json').write_text(json.dumps(selected_ids), encoding='utf-8')
        meta = dict(packed.meta)
        meta.update(count=len(selected_ids), dtype='float16', restricted_to_chunk_ids=True)
        (store.packed_dir / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    vectors = load_packed_vectors(store.packed_dir)
    stamp = datetime.now(timezone.utc).isoformat()
    snapshot = build_corpus_snapshot(chunks, created_at=stamp)
    if snapshot.chunk_count != len(chunks):
        raise ValueError('Every selected chunk must have a document ID.')
    write_corpus_snapshot(store, snapshot)
    files = {path.relative_to(output).as_posix(): file_hash(path)
             for path in sorted((output / 'data').rglob('*')) if path.is_file()}
    manifest = {'schema_version': 1, 'built_at': stamp, 'snapshot_id': snapshot.snapshot_id,
                'tickers': sorted(wanted), 'source_company_count': len(source_tickers),
                'chunk_count': len(chunks), 'vector_count': len(vectors),
                'missing_vector_count': len(ids - set(vectors)),
                'latest_filing_date': max(str(c.metadata.get('filing_date', '')) for c in chunks),
                'files': files}
    # Completion marker is written last; partial builds cannot start the hosted app.
    (output / 'demo_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tickers-file', type=Path, default=ROOT / 'config/demo_tickers.txt')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/demo_slice')
    parser.add_argument('--packed-source', type=Path, help='Optional existing pack from the same frozen corpus.')
    args = parser.parse_args()
    tickers = [line.strip().upper() for line in args.tickers_file.read_text(encoding='utf-8').splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
    print(json.dumps(build_slice(source=args.root, output=args.out, tickers=tickers,
                                 packed_source=args.packed_source), indent=2))


if __name__ == '__main__':
    main()
