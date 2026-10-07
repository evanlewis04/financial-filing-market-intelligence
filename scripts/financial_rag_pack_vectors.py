"""Pack the per-chunk vector cache into one memory-mapped matrix.

``data/vector_cache`` holds one JSON file per chunk (265,742 files on the
503-company corpus), which takes ~3 minutes and several GB of RAM to load. This
writes ``data/packed/{vectors.npy,chunk_ids.json,meta.json}``, which
``load_local_retrieval_corpus`` picks up automatically on the next run.

Rebuild to a fresh --out directory after embedding new filings; the per-chunk cache stays the source of
truth, so the packed artifact is always rebuildable and safe to delete.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.retrieval import pack_vector_cache
from src.financial_rag.settings import project_root
from src.financial_rag.storage import LocalRagStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack cached chunk vectors into a single matrix.")
    parser.add_argument(
        "--out",
        default="",
        help="Output directory (default: data/packed under the project root).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Stored matrix dtype; float16 halves storage but may reorder close dense scores.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = LocalRagStore(root=project_root())
    out_dir = Path(args.out) if args.out else store.packed_dir
    started = time.perf_counter()
    meta = pack_vector_cache(
        vector_cache_dir=store.vector_cache_dir,
        out_dir=out_dir,
        dtype=args.dtype,
    )
    elapsed = time.perf_counter() - started
    print(f"Packed {meta['count']} vectors ({meta['dimensions']} dims, {meta['dtype']}) in {elapsed:.1f}s")
    print(f"  out: {out_dir}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
