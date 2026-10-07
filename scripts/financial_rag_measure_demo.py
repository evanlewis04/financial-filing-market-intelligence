"""Measure the application's uncached demo service in a fresh process."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil

from scripts.financial_rag_brief_view import _demo_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'artifacts/demo_slice')
    parser.add_argument('--cache-state', required=True, help='Record observed OS cache state; do not assume cold.')
    parser.add_argument('--use-voyage', action='store_true')
    args = parser.parse_args()
    service, manifest = _demo_service(str(args.root), args.use_voyage)
    seconds = time.perf_counter() - START
    memory = psutil.Process().memory_info()
    if sys.platform == 'win32':
        peak = memory.peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    print(json.dumps({'platform': platform.platform(), 'python': platform.python_version(),
                      'cache_state': args.cache_state, 'fresh_process': True, 'corpus_sharing': False,
                      'seconds_including_imports': seconds, 'peak_rss_bytes': peak,
                      'current_rss_bytes': memory.rss, 'health': service.health(),
                      'snapshot_id': manifest['snapshot_id'],
                      'acceptance_met': seconds < 5 and peak < 1_000_000_000}, indent=2))


if __name__ == '__main__':
    main()
