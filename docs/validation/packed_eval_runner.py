import json
import os
import runpy
import sys
from pathlib import Path

if "--execute" not in sys.argv:
    print("Frozen inputs ready; launch evaluation separately with --execute to give it a full tool time window.", flush=True)
    raise SystemExit(75)

root, evidence, label = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
sys.path.insert(0, str(root))
os.chdir(root)
import src.financial_rag.evaluation as evaluation
import time
import psutil
from src.financial_rag import retrieval
load_original = retrieval.load_local_retrieval_corpus
load_count = 0
loaded_corpora = {}

def measure_load(*args, **kwargs):
    global load_count
    started = time.perf_counter()
    key = (str(Path(kwargs.get("root", ".")).resolve()), repr(kwargs.get("snapshot")))
    reused = key in loaded_corpora
    if not reused:
        loaded_corpora[key] = load_original(*args, **kwargs)
    result = loaded_corpora[key]
    elapsed = time.perf_counter() - started
    load_count += 1
    mem = psutil.Process().memory_info()
    stats = dict(reused_frozen_corpus=reused, seconds=elapsed, rss_bytes=mem.rss, peak_rss_bytes=getattr(mem, "peak_wset", mem.rss),
                 chunks=len(result[0]), vectors=len(result[1]), vector_type=type(result[1]).__name__)
    (evidence / f"{label}_load_{load_count}.json").write_text(json.dumps(stats, indent=2))
    print("LOAD", stats, flush=True)
    return result

retrieval.load_local_retrieval_corpus = measure_load
# The evaluator reads the same frozen corpus twice. Share the returned objects
# to avoid keeping two multi-GB copies alive; neither gold resolution nor the
# query service mutates them. Apply identically to both source trees.
import src.financial_rag.api.local_service as api_service
api_service.load_local_retrieval_corpus = measure_load
original = evaluation.build_retrieval_quality_report

def capture(cases, payloads, **kwargs):
    (evidence / f'{label}_payloads.json').write_text(json.dumps(payloads, sort_keys=True), encoding='utf-8')
    (evidence / f'{label}_cases.json').write_text(json.dumps([c.to_dict() for c in cases], sort_keys=True, default=str), encoding='utf-8')
    return original(cases, payloads, **kwargs)

evaluation.build_retrieval_quality_report = capture
sys.argv = ['financial_rag_expanded_retrieval_eval.py', '--top-k', '5', '--per-subquery-k', '8', '--reranker', 'none', '--output', str(evidence / f'{label}_eval.json'), '--csv-output', str(evidence / f'{label}_eval.csv')]
print('ARGV', sys.argv, flush=True)
runpy.run_path(str(root / 'scripts/financial_rag_expanded_retrieval_eval.py'), run_name='__main__')
