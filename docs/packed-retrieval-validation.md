# Packed retrieval validation

This change adds a derived, memory-mapped vector cache and preserves the dict
constructor and per-file fallback. It also fixes the dense score attached to
each result; the old implementation returned the last candidate's dense score.
Lexical scoring, fusion weight (0.25), evaluator, cases, and gold-label resolver
are unchanged.

## Storage and numerical contract

Run `python scripts/financial_rag_pack_vectors.py` to create `data/packed`.
The output consists of `vectors.npy`, ordered `chunk_ids.json`, and `meta.json`.
All three must exist before readers select the packed path. Metadata shape and
dtype are checked when opening the pack; invalid artifacts fail explicitly.
JSON remains the source of truth. Packing refuses to overwrite an existing
artifact: rebuild into a fresh `--out` directory, then replace the old derived
cache while readers are stopped. Remove the derived cache to use JSON again.
Rebuild after embedding new filings; packs do not automatically track ingestion.

Packing preserves sorted traversal and the JSON loader's last-file-wins rule
for duplicate chunk IDs. Mixed dimensions/models, nonfinite values, and values
outside the selected dtype's range fail packing rather than dropping evidence.
The optional chunk-id allowlist supports the later hosted subset.

Float16 is the default; `--dtype float32` is supported. Computation uses float32
candidate rows and query vectors, in batches of at most 4,096 candidate rows
to bound temporary allocations for unfiltered queries. Tests compare 1024-dimensional matmul cosine
against pure Python on the same quantized vectors with absolute tolerance
`2e-7`, and against original normalized vectors with tolerance `1e-4`. These
are tested bounds on seeded fixtures, not a universal error guarantee. Other
ranking fixtures use `5e-3` dense / `2e-3` combined score tolerances and require
identical ordering. Missing, zero, and dimension-mismatched vectors score zero.

Exact dense ranking equivalence is **not** promised: float16 can collapse nearby
scores into a tie. A dedicated regression demonstrates that case. Stable ties
retain input corpus order. The controlled lexical comparison requires exact
ordered results and metrics; it does not validate live Voyage retrieval quality.

## Historical provenance

The July 31 README baseline describes a 12-ticker, 6,259-chunk corpus and 58
resolved gold labels. The saved October 1 log describes 268,181 chunks, 265,741
vectors and 64 resolved labels at unmodified `5d5f6e1`. Its launcher used
`--reranker none` and default `--per-subquery-k 5`; the required comparison uses
`--top-k 5 --per-subquery-k 8`. The old run has no input hashes or ordered result
payloads, so it cannot establish equivalence for this change.

The historical 0.860 / 0.667 / 0.421 versus October 1
0.980 / 0.386 / 0.170 discrepancy remains unresolved. Differences in corpus and
arguments are recorded provenance differences, not a demonstrated explanation.
Additional readily available provenance: commit `4010e52` (PR #6, August 1)
replaced gold selection based on the production retrieval scorer with an
independent term-count signal. The evaluation history documents the same-corpus
Recall/MRR moving from 0.667/0.421 to 0.436/0.194 after that correction. The
subsequent pinned-filing update documents 64 labels and 0.409/0.177. These
already-recorded changes make the July README numbers an unsuitable current
acceptance reference; they do not fully attribute the October result.
Neither historical numbers nor gold labels are changed by this PR.

## Controlled comparison

The completed comparison is recorded in
[the machine-readable evidence](packed-retrieval-comparison.json). Source inputs are copied into an isolated reference
root and individually SHA-256 hashed. Unmodified code comes from a Git archive
of `5d5f6e1`; the after root overlays the reviewed changes onto that archive and
shares the frozen input directories. Both use the same Python environment and
arguments. An external runner captures the inputs to the unchanged report
builder (cases with resolved gold IDs and complete per-case API payloads) and
times the actual first corpus load. The runner memoizes the repeated read of
the same frozen root for gold-label resolution and service construction,
identically on both sides. Both consumers only read the shared objects. This
avoids retaining two corpus copies; it is a disclosed harness adjustment, not
a benchmark of stock whole-evaluator memory. The stock duplicate-load attempt
was stopped after severe paging at 15.3 GB private memory, and its log retained.
The measured first load still invokes the unmodified reference loader or the
changed packed loader, respectively.

Deploy remains gated on a real-Voyage evaluation and retrieval-quality review,
plus the hosted subset's load and cited-query acceptance checks.


## Results — October 6, 2026

Both runs completed all 50 cases and exited zero. Per-case ordered chunk IDs,
complete API payloads, cases with resolved gold IDs, and complete evaluation
reports are exactly equal. Both report source hit **0.980**, Recall@5 **0.386**,
MRR **0.170**, NDCG@5 **0.187**, 64 resolved labels, and the one intended
unsupported-ticker control. SHA-256 checks confirm the evaluator/case/gold source
files are unchanged. The measured after export matches all 100 Python files in
the worktree that passed verification.

| First corpus load | Seconds | Peak working set (decimal GB) |
| --- | ---: | ---: |
| Unmodified JSON loader | 582.116 | 9.226 |
| Packed vectors plus existing chunks | 22.299 | 1.920 |
| Packed vector store alone (diagnostic) | 0.230 | 0.101 |

**The full-corpus targets are not met:** 22.3 s exceeds 10 s and 1.92 GB exceeds
1.5 GB. Vector packing addresses the vector-cache cost; the existing chunk
text/metadata loader remains the next performance constraint. Planner removed
these full-corpus targets from PR A merge acceptance on October 6, 2026
(Buzz decision `955f89b52c4c431c55b26e509a413b2e1c81cd3c19349e739f57aede71e5d33e`;
workspace `PLANS/FILINGS_PERFORMANCE_SCOPE_2026_10_06.md`). They remain unmet
goals; full-corpus chunk-loader optimization is deferred. This is neither merge
approval nor a signing-policy waiver. Validator reviewed `1cd1b29` and reported
no additional code defect. PR B retains <5 s / <1 GB subset acceptance through
the actual application load path in a fresh process without evaluator sharing. These are local Windows
measurements on a 16.8 GB machine under memory pressure, not portable latency
or speedup guarantees. The end-of-load RSS on the JSON path was depressed by
paging, so the table uses the recorded OS peak working set for both paths.

The pack contains 265,741 vectors of 1,024 dimensions, model `voyage-finance-2`,
with no duplicate IDs. `vectors.npy` is 544,237,696 bytes and the ordered ID file
is 26,054,830 bytes. Packing is a separate one-time build step (547.3 s in this
run); startup measurement excludes it.

Verification: `scripts/verify.py` completed lint, compile, **171 tests**, and
brief smoke. The tests ran in the changed worktree based on `5d5f6e1`; source
hashes bind that worktree to the measured after export. The 4,096-row batching
bounds dense temporary allocations, including an explicit boundary-alignment
regression. The chunk-id allowlist is the approved substitute for a ticker
parameter; ticker selection belongs with chunk metadata in the later slice tool.

Evidence is retained in the workspace at
`RESEARCH/FILINGS_CONTROLLED_2026_10_06/`, including the dependency freeze, code
hashes, input manifest, complete before/after payloads, logs, and packed artifact.
The full corpus is not committed. Its manifest contains 5,435 chunk JSONL files
(849,855,295 bytes) and 265,741 vector JSON files (7,644,565,205 bytes). The
manifest SHA-256 is
`02fccd9fe2b50bc186352f52d2dc6f66316872b230c9fcb43ebbc15f8328fcb1`.

The exact external runner is included at
[`validation/packed_eval_runner.py`](validation/packed_eval_runner.py). After
preparing reference and changed source roots with identical frozen inputs, run:

```bash
python docs/validation/packed_eval_runner.py REFERENCE_ROOT EVIDENCE_DIR before --execute
python docs/validation/packed_eval_runner.py CHANGED_ROOT EVIDENCE_DIR after --execute
```

It captures results around the existing evaluator. Its memoization applies only
to this measurement process; no runtime corpus caching was added to the product.

Post-run integrity: all **271,176 files** matched their recorded byte hashes,
and the file inventory was unchanged after both evaluations.
