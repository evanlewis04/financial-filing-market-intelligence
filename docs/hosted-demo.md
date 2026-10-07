# Pinned hosted demo

The hosted entrypoint is `deploy/streamlit_app.py`. It requires `APP_PASSWORD`
before loading the corpus or exposing query/answer controls. The existing local
entrypoint remains available. Provider keys are bridged from server-owned
Streamlit secrets into the existing SDK environment. Missing Voyage configuration
is labeled **lexical only**; provider request failures show a retry message.
Generated answers remain opt-in behind the password and existing evidence gates.
Restart the app after changing provider secrets so cached clients are rebuilt.

## Artifact and coverage

`config/demo_tickers.txt` proposes 50 recognizable companies across 11 sectors.
The prepared slice under `artifacts/demo_slice/` contains all cached filings for
those tickers: **26,354 chunks, 25,582 float16 vectors, 772 chunks without vectors**.
Missing vectors remain visible and retain lexical scoring; no coverage was dropped
to meet performance targets. The selected artifact is 135,933,591 bytes, including
52,392,064 bytes for the vector matrix. No individual file exceeds 100 MB.

Composition is pinned using the existing corpus snapshot mechanism:
`sha256:1ca318c228b8087fcc2e4c6fc8fb30cb8bb4c17746bad03118e734feae2ecd41`.
The artifact was built October 6, 2026 from the frozen full corpus; its newest
cached filing is August 5, 2026. Build date is not a claim of current SEC coverage.
The UI reports these dates separately and says 50 of 503 companies.

`demo_manifest.json` is the completion marker and records every artifact file's
SHA-256. Startup checks the complete inventory, hashes, snapshot and counts before
serving data. Incomplete or changed artifacts fail closed. The manifest, UI
options and per-service API universe agree. Legacy API defaults are unchanged.
All 50 tickers returned cited, ticker-matching evidence in a basic offline check;
this is a coverage check, not a quality evaluation for all 50 companies.
IBM, present in the full corpus but outside this slice, was rejected with
`unsupported_ticker` before answer generation (zero synthesis calls).

Build into a fresh directory from a local corpus; no fetch or embedding calls:

```bash
python scripts/financial_rag_build_demo_slice.py --root /path/to/corpus --tickers-file config/demo_tickers.txt --out artifacts/demo_slice
```

Optionally pass `--packed-source /path/to/data/packed` for a validated pack from
the **same frozen corpus**. The recorded build used the PR A pack with matrix
SHA-256 `ec4985cb87b1580a3bb757e5c2c23d8748906826274983b7af365415c700626b`.
Parent input manifest SHA-256:
`02fccd9fe2b50bc186352f52d2dc6f66316872b230c9fcb43ebbc15f8328fcb1`.
The builder preserves selected input-file and chunk order, reuses the snapshot
writer, and refuses to overwrite output. Failed builds have no completion marker;
use a new directory for a retry. Review the artifact diff before replacing a pin.

## Local acceptance evidence

Measurements use `scripts/financial_rag_measure_demo.py`, which imports the actual
Streamlit view and invokes its cached-service function in a **fresh process with
an empty resource cache**. Timing includes imports, integrity checks, chunks,
vectors and service initialization. No evaluator corpus-sharing hook is used.

| Windows 11 / Python 3.13.5 | Startup | Peak process RSS |
| --- | ---: | ---: |
| Existing development environment, warm OS cache | 2.498 s | 267,739,136 bytes |
| Clean slim deployment environment, warm OS cache | 2.109 s | 247,320,576 bytes |

Both meet the local `<5 s / <1 GB` subset targets. Neither is a cold-disk or Linux
host measurement. Host acceptance remains open. Streamlit caches the validated
read-only service across reruns; password checks run before retrieving that cache.

```bash
python scripts/financial_rag_measure_demo.py --cache-state "describe the actual OS cache state"
```

Full `scripts/verify.py` on the changed worktree based on `1cd1b29`: **186 passed,
2 skipped**, lint, compilation and brief smoke passed. An isolated venv installed
only `deploy/requirements.txt` successfully and exercised app startup and the
universe checks. Direct runtime dependencies are pinned in `requirements-deploy.txt`.
Local Streamlit AppTest exercised wrong password, login, 50 choices, a real
Voyage/OpenAI generated answer with five accepted citations, and sign-out relocking.
This is local application evidence, not a browser or deployed-host acceptance claim.

Machine-readable summaries are in `docs/validation/demo-*.json`; complete logs,
payloads, source-review packet and source hashes are in workspace
`RESEARCH/FILINGS_DEMO_2026_10_06/`.

## Real-Voyage quality review

The unchanged 50-case evaluator covers **13 companies**, including its historical
TSLA negative control. It does not quality-validate every company in the slice.
Both runs use this snapshot, `top-k=5`, `per-subquery-k=8`, no reranker, unchanged
gold labels and the existing dense fusion weight 0.25. Resolved cases are identical.

| Metric | Lexical | Real Voyage + lexical |
| --- | ---: | ---: |
| Section/source hit | 1.000 | 1.000 |
| Recall@5 | 0.386 | 0.386 |
| MRR | 0.170 | 0.209 |
| NDCG@5 | 0.187 | 0.213 |

The TSLA expectation is **stale and remains a recorded failure**, not a passed
negative control: TSLA is intentionally supported by this demo. The separate IBM
check validates rejection. The source-hit increase from PR A's 0.980 reflects
that universe change, not improved retrieval quality.

Three cases regress: META advertising loses both resolved gold IDs; AAPL gross
margin's gold passage moves from rank 2 to 3; NVDA press-release gross margin
moves from rank 4 to 5. NVDA inventory gains a gold hit, offsetting META in mean
Recall. The four-case packet includes complete source passages, dates, filters,
gold text and top-five citations for independent review. Float16 limitations from
PR A still apply; no weight, label, case or default-mode tuning was performed.

Planner's October 6 disposition (`332a3c881d7fa68742a57d0e9c72fdc4adc4e2e34caa91b102286c609e426194`)
accepts these bounded-demo limitations **subject to Validator's final source/filter
findings**. META's broad question retains direct advertising support but loses
quantitative growth/seasonality/FX context. AAPL claims must identify their period.
NVDA retains the same five margin passages. This does not authorize unsupported
quantitative claims. Final review and host acceptance remain required.

## Deployment handoff (after review/signing requirements are resolved)

1. Connect the repository to Streamlit Community Cloud and choose the reviewed
   branch/commit, entrypoint `deploy/streamlit_app.py`, and Python 3.13.
2. Set `APP_PASSWORD`, `OPENAI_API_KEY`, and `VOYAGE_API_KEY` in the app's secrets
   settings. Use real private values there, never in git or chat. No secrets file
   is shipped with this artifact. Missing password leaves the hosted app locked.
3. Confirm the pinned snapshot and 50-company banner. Measure a fresh host process
   against `<5 s / <1 GB`, recording platform, cold/warm state and peak RSS.
4. Exercise wrong password, login, cited briefs, an opt-in generated answer with
   validated citations, IBM rejection without generation, and sign-out. Confirm
   lexical-only labeling when Voyage is absent; restore the required key afterward.
5. Record the live URL only after acceptance; README live-link/screenshot work is
   the later PR C. The app has not been deployed as part of PR B.

Community Cloud searches dependencies beside the entrypoint before the repository
root, so `deploy/requirements.txt` includes the slim root file. See official
[dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
and [secrets settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

Full-corpus `<10 s / <1.5 GB` goals remain unmet at 22.3 s / 1.92 GB and were
explicitly deferred by Planner's performance scope decision. No full-corpus
chunk-storage redesign is included here. Deployment/signing approval is separate.

### Blocking source review finding

Validator completed the four-case source/filter review (Buzz event
`13f62ff0e95c88ee56b181c50e8e589aa91bdbac3469a8b957f9256dbb03a9ed`).
Full-chunk evidence passes all four cases with the documented limitations, but
AAPL generated-answer support is blocked: synthesis only sees 320-character
`source_excerpt` prefixes, which omit the retrieved margin-driver passages.
The local successful answer/citation smoke does not establish claim support for
these omitted passages. The complete demo quality gate remains open pending a
separately scoped evidence-delivery fix and offline prompt-content validation.
No further paid runs or retrieval tuning are needed for that review.

### Evidence-delivery implementation

PR B now resolves exactly the selected chunk IDs against the service's loaded
pinned corpus before synthesis. An explicit full-evidence input carries complete
text and identity metadata; the 320-character UI excerpts and retrieval/evaluation
payloads remain unchanged. Missing, duplicate or mismatched records block the
answer before a provider call. The prompt preserves citation labels, order,
filing dates, report dates, period ends, accession and URLs, and instructs answers
to distinguish reporting periods and results from outlook.

There was no explicit model input budget in the prior synthesis code. This fix
uses a conservative **18,000-character application limit** over instructions plus
input, chosen for this demo,
and at most 20 selected sources. Planner accepted this new application policy
in decision `b1a9167426778ed33fd92f0044ea2390368d97d7937fb1ff803e2c57448acbf8`.
The former 20 x 900-character excerpt cap was not a total-prompt budget;
characters are not tokens or a cost guarantee. This is not the provider context-window size.
Oversize yields a clear blocked answer state with no silent truncation. Legacy
low-level synthesis callers without full evidence remain explicitly excerpt-only.

Offline replay through the actual brief path and the loaded pinned service
captured complete final prompts for all four reviewed cases, with no paid calls.
AAPL costs/mix/tariffs and all text-to-citation/date mappings are present. Prompt
sizes are 12,189 / 12,797 / 10,668 / 11,217 characters respectively for META,
AAPL, NVDA margin, and NVDA inventory. See `demo-prompt-capture.json` and the
workspace `*_PROMPT.json` files. Missing, ambiguous, mismatched and oversized
source tests verify zero provider calls. Validator review remains the gate;
these tests do not assert the quality of a newly generated model answer.

### Evidence gate closure and remaining gates

Planner closed the four-case source-support and full-text delivery gate for this
snapshot/configuration in decision
`abcbc2b872f524291012ca777f192ddea6ba16527fe1ca5acfe0d36cd3a94ab3`, based on
Validator's independent verdict
`380dc28307e0fb2098c5fb247494397012c8ccb20a36de670456859b260b7d5d`.
This supersedes the pending-review status above for those specific findings.
It confirms supporting evidence reaches synthesis; it does not certify generated
prose accuracy. The META limitation and 13-company quality scope remain.

Additional offline checks verify acceptance at exactly 18,000 input characters,
blocking at 18,001 with no additional provider call, and the actual hosted UI
showing the blocked state and reduce-Top-k/narrow-question guidance without a raw
exception. Final verification reports 186 passed and two skipped tests.

Artifact paths are marked `-text` in `.gitattributes` so Git cannot rewrite line
endings and invalidate the pin on Linux. All 535 staged data-file blobs match
the manifest hashes. The manifest itself is also committed byte-for-byte.

Final PR B review, compliant commit signing and eventual host acceptance remain
open. The configured `git-sign-nostr` helper is absent from the current Buzz
installation; further commits are pending repair of that runtime, without a
signing bypass. PR B is staged in its worktree and has not been published yet.

Final-review rebuild correction: the JSON-vector builder now passes the packer
directories by keyword. A regression builds without `packed_source`, loads the
manifest/snapshot, checks vector values, excludes unrelated vectors, and retains
missing-vector chunks. Full verification: 186 passed, two skipped. The supplied
artifact and snapshot are unchanged.
