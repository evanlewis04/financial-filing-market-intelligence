"""Recruiter-facing analyst brief: one screen pairing cited SEC filing evidence
with an offline options-market snapshot, plus an opt-in grounded answer.

This is the single UI surface for the project. It renders an Apple-inspired
design system (CSS-variable tokens with a runtime light/dark toggle) entirely
inline, so the view stays self-contained after the dashboard/quant stack was
retired. The market snapshot is deterministic and offline by default and is
always labeled as context, never as filing evidence.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.financial_rag.api import build_local_api_service
from src.financial_rag.integration import (
    build_unified_brief,
    market_provider_from_metrics,
)
from src.financial_rag.hosted_demo import (
    bridge_provider_secrets, load_demo_service, require_password, retrieval_notice,
)
from src.financial_rag.settings import project_root
from src.financial_rag.workbench import company_options

# --- Apple-inspired design system -------------------------------------------
# Single token source. Light and dark share the same component CSS; only the
# `:root` custom-property block differs, so a runtime toggle just re-injects the
# other token set on rerun (no client-side JS, robust inside Streamlit).

_TOKENS_LIGHT = """
:root {
  --bg: #F5F5F7; --panel: #FFFFFF; --panel-2: #FBFBFD;
  --ink: #1D1D1F; --muted: #6E6E73; --muted-2: #86868B;
  --line: #D2D2D7; --line-soft: #E8E8ED;
  --accent: #0071E3; --accent-soft: rgba(0,113,227,0.10);
  --chip-filing-bg: rgba(0,113,227,0.10); --chip-filing-ink: #0060C7;
  --chip-market-bg: rgba(142,142,147,0.14); --chip-market-ink: #4B4B50;
  --pos: #1D7A3E; --neg: #C1121F;
  --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06);
  --shadow-lift: 0 2px 6px rgba(0,0,0,0.06), 0 18px 48px rgba(0,0,0,0.10);
}
"""

_TOKENS_DARK = """
:root {
  --bg: #000000; --panel: #1C1C1E; --panel-2: #161618;
  --ink: #F5F5F7; --muted: #A1A1A6; --muted-2: #8A8A8F;
  --line: #2C2C2E; --line-soft: #242426;
  --accent: #0A84FF; --accent-soft: rgba(10,132,255,0.16);
  --chip-filing-bg: rgba(10,132,255,0.18); --chip-filing-ink: #6AB3FF;
  --chip-market-bg: rgba(142,142,147,0.20); --chip-market-ink: #C4C4C8;
  --pos: #4ED07A; --neg: #FF6B6B;
  --shadow: 0 1px 2px rgba(0,0,0,0.5), 0 10px 30px rgba(0,0,0,0.5);
  --shadow-lift: 0 2px 8px rgba(0,0,0,0.6), 0 22px 56px rgba(0,0,0,0.62);
}
"""

_COMPONENT_CSS = """
:root {
  --font-ui: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", "Inter", Arial, sans-serif;
  --font-mono: "SF Mono", "JetBrains Mono", "Cascadia Mono", ui-monospace, Consolas, monospace;
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px; --r-xl: 20px;
}

.stApp { background: var(--bg); color: var(--ink); font-family: var(--font-ui);
  transition: background 300ms ease, color 300ms ease; }
.block-container { max-width: 1140px; padding-top: 2.2rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: var(--ink); font-family: var(--font-ui);
  font-weight: 600; letter-spacing: -0.015em; }
p, span, label, li, div { font-family: var(--font-ui); }

/* Sidebar chrome */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line-soft); }
section[data-testid="stSidebar"] * { color: var(--ink); }
[data-testid="stExpander"] { border: 1px solid var(--line-soft); border-radius: var(--r-md);
  background: var(--panel-2); }

/* Inputs — kept coherent in both light and dark via the token variables */
textarea, input { border-radius: var(--r-sm) !important; font-family: var(--font-ui) !important; }
.stTextArea textarea, .stTextInput input,
[data-testid="stNumberInput"] input, [data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
  background: var(--panel-2) !important; color: var(--ink) !important;
  border-color: var(--line) !important; }
[data-testid="stNumberInput"] button {
  background: var(--panel-2) !important; color: var(--ink) !important;
  border-color: var(--line) !important; }
/* Selectbox: 1.60 renders the control on a dynamic emotion-class grandchild,
   so target it structurally via the stable stSelectbox testid. */
[data-testid="stSelectbox"] > div > div {
  background: var(--panel-2) !important; color: var(--ink) !important;
  border-color: var(--line) !important; border-radius: var(--r-sm) !important; }
[data-testid="stSelectbox"] svg, [data-testid="stExpander"] summary svg { fill: var(--muted) !important; }
[role="listbox"] { background: var(--panel) !important; border: 1px solid var(--line) !important; }
[role="option"] { color: var(--ink) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--ink) !important; }

/* Primary button */
.stButton > button { border-radius: 980px; font-weight: 500; font-family: var(--font-ui);
  border: 1px solid var(--line); transition: transform 200ms ease, box-shadow 200ms ease; }
.stButton > button[kind="primary"] { background: var(--accent); color: #FFFFFF;
  border: none; box-shadow: var(--shadow); padding: 0.5rem 1.4rem; }
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: var(--shadow-lift); }

/* ---- Hero ---- */
.hero { margin: 4px 0 34px; max-width: 780px; }
.hero .eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 11.5px;
  font-weight: 600; color: var(--accent); margin-bottom: 12px; }
.hero h1 { margin: 0 0 14px; font-size: 44px; line-height: 1.06; font-weight: 600;
  letter-spacing: -0.026em; color: var(--ink); }
.hero .lede { margin: 0; font-size: 19px; line-height: 1.45; color: var(--muted);
  font-weight: 400; letter-spacing: -0.006em; }
.hero .lede strong { color: var(--ink); font-weight: 500; }

/* ---- Card ---- */
.card { background: var(--panel); border: 1px solid var(--line-soft); border-radius: var(--r-xl);
  box-shadow: var(--shadow); padding: 24px 24px 22px; margin-bottom: 20px; }
.card-head { display: flex; align-items: flex-start; justify-content: space-between;
  margin-bottom: 18px; gap: 12px; }
.card-title { font-size: 17px; font-weight: 600; letter-spacing: -0.014em; margin: 0; color: var(--ink); }
.card-sub { font-size: 13px; color: var(--muted); margin: 4px 0 0; }

.chip { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap;
  font-size: 11px; font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;
  padding: 5px 11px; border-radius: 980px; }
.chip .dot { width: 6px; height: 6px; border-radius: 50%; }
.chip.filing { background: var(--chip-filing-bg); color: var(--chip-filing-ink); }
.chip.filing .dot { background: var(--chip-filing-ink); }
.chip.market { background: var(--chip-market-bg); color: var(--chip-market-ink); }
.chip.market .dot { background: var(--chip-market-ink); }

/* ---- Evidence rows ---- */
.evidence { display: flex; flex-direction: column; gap: 12px; }
.ev { border: 1px solid var(--line-soft); border-radius: var(--r-md); background: var(--panel-2);
  padding: 14px 16px; transition: box-shadow 200ms ease, border-color 200ms ease; }
.ev:hover { box-shadow: var(--shadow-lift); border-color: var(--line); }
.ev-top { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.cite { font-family: var(--font-mono); font-size: 11.5px; font-weight: 600;
  background: var(--accent-soft); color: var(--accent); padding: 2px 8px; border-radius: 6px; }
.meta { font-size: 12px; color: var(--muted-2); font-variant-numeric: tabular-nums; }
.meta b { color: var(--muted); font-weight: 500; }
.ev-text { font-size: 13.5px; line-height: 1.5; color: var(--ink); margin: 0 0 8px; }
.ev-src { font-size: 12px; color: var(--accent); text-decoration: none; word-break: break-all; }
.ev-src:hover { text-decoration: underline; }

/* ---- Market metrics ---- */
.metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 4px; }
.metric { background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: var(--r-md);
  padding: 14px 15px; }
.metric .k { font-size: 11.5px; color: var(--muted); font-weight: 500; margin-bottom: 6px;
  letter-spacing: 0.01em; }
.metric .v { font-size: 26px; font-weight: 600; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; font-family: var(--font-mono); color: var(--ink); }
.metric .v.pos { color: var(--pos); }
.metric .v.neg { color: var(--neg); }
.snapshot-note { margin-top: 16px; font-size: 12px; color: var(--muted);
  background: var(--chip-market-bg); border-radius: var(--r-sm); padding: 10px 12px; line-height: 1.45; }
.snapshot-note b { color: var(--ink); font-weight: 600; }

/* ---- Answer + generic prose ---- */
.answer-text { font-size: 14.5px; line-height: 1.6; color: var(--ink); margin: 0 0 8px; white-space: pre-wrap; }
.note-list { margin: 6px 0 0; padding-left: 18px; color: var(--muted); font-size: 13px; line-height: 1.6; }
.subtle { font-size: 12.5px; color: var(--muted-2); font-variant-numeric: tabular-nums; }

/* ---- Data-source rows ---- */
.src-row { display: flex; align-items: center; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--line-soft); font-size: 13px; }
.src-row:last-child { border-bottom: 0; }
.src-row .lbl { font-family: var(--font-mono); font-size: 12px; color: var(--ink); min-width: 190px; }
.src-row .prov { color: var(--muted); }

/* ---- States ---- */
.empty { background: var(--panel); border: 1px dashed var(--line); border-radius: var(--r-xl);
  padding: 30px 28px; text-align: left; box-shadow: none; }
.empty h3 { margin: 0 0 6px; font-size: 18px; }
.empty p { margin: 0 0 18px; color: var(--muted); font-size: 14px; max-width: 620px; line-height: 1.55; }
.empty .steps { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.empty .step { border: 1px solid var(--line-soft); border-radius: var(--r-md); background: var(--panel-2);
  padding: 14px 16px; }
.empty .step .st-t { font-weight: 600; font-size: 13.5px; margin-bottom: 4px; color: var(--ink); }
.empty .step .st-d { color: var(--muted); font-size: 12.5px; line-height: 1.45; }
.warn { border: 1px solid var(--neg); border-radius: var(--r-md); background: var(--chip-market-bg);
  color: var(--ink); padding: 12px 14px; font-size: 13px; }
"""


def _theme_css(dark: bool) -> str:
    tokens = _TOKENS_DARK if dark else _TOKENS_LIGHT
    return f"<style>{tokens}{_COMPONENT_CSS}</style>"


def inject_theme(st_module: Any, dark: bool = False) -> None:
    """Apply the Apple-inspired token theme (light by default) to the app."""
    st_module.markdown(_theme_css(dark), unsafe_allow_html=True)


DETERMINISTIC_SNAPSHOT = {
    "source_mode": "Fallback",
    "message": "Deterministic offline market snapshot (not live).",
    "front_expected_move_pct": 8.2,
    "iv_rank": 64.0,
    "iv_30d": 0.52,
    "skew": -0.04,
}

# Presentation for known market metrics: (label, formatter, signed?).
_METRIC_SPEC: dict[str, tuple[str, Callable[[float], str], bool]] = {
    "front_expected_move_pct": ("Front expected move", lambda v: f"{v:.1f}%", False),
    "iv_rank": ("IV rank", lambda v: f"{v:.0f}", False),
    "iv_30d": ("30-day IV", lambda v: f"{v:.2f}", False),
    "skew": ("Skew", lambda v: f"{v:+.2f}".replace("-", "−"), True),
}


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _short_url(url: str) -> str:
    """Compact an EDGAR archive URL for display without losing the filename."""
    if not url:
        return ""
    tail = url.split("/Archives/", 1)[-1] if "/Archives/" in url else url
    parts = tail.split("/")
    if len(parts) > 2:
        return "sec.gov/…/" + _esc(parts[-1])
    return _esc(url)


def _hero_html(ticker: str) -> str:
    return (
        '<div class="hero">'
        f'<div class="eyebrow">{_esc(ticker)} · SEC filings</div>'
        "<h1>SEC filing intelligence,<br/>with the citation attached.</h1>"
        '<p class="lede">Ask a question, get <strong>cited evidence from the filings '
        "themselves</strong> — paired with an offline options-market snapshot, and "
        "fully reproducible run-to-run.</p>"
        "</div>"
    )


def _empty_state_html() -> str:
    return (
        '<div class="card empty">'
        "<h3>Build a cited filing brief</h3>"
        "<p>Pick a ticker and a question, then <b>Build Brief</b>. You get management's own "
        "disclosure with validated citations, an offline market snapshot for context, and an "
        "optional grounded answer — nothing is shown that isn't traceable to a source.</p>"
        '<div class="steps">'
        '<div class="step"><div class="st-t">SEC filing — cited</div>'
        '<div class="st-d">Passages pulled straight from EDGAR filings, each with an accession '
        "and source link. This is the evidence.</div></div>"
        '<div class="step"><div class="st-t">Offline snapshot</div>'
        '<div class="st-d">A deterministic options-market context panel — labeled clearly as '
        "context, never treated as filing evidence.</div></div>"
        '<div class="step"><div class="st-t">Opt-in answer</div>'
        '<div class="st-d">A grounded answer is generated only if you ask, and only from cited '
        "evidence. Hallucinated citations are dropped.</div></div>"
        '<div class="step"><div class="st-t">Reproducible</div>'
        '<div class="st-d">Eval-critical filings are pinned by accession, so the same run '
        "reproduces on a fresh clone.</div></div>"
        "</div></div>"
    )


def _evidence_card_html(evidence: list[dict[str, Any]]) -> str:
    rows = []
    for item in evidence:
        meta = (
            f'<b>{_esc(item.get("ticker", ""))}</b> · {_esc(item.get("form_type", ""))} · '
            f'<b>{_esc(item.get("filing_date", ""))}</b>'
        )
        url = str(item.get("source_url", "") or "")
        src = (
            f'<a class="ev-src" href="{_esc(url)}" target="_blank" rel="noopener">{_short_url(url)}</a>'
            if url
            else ""
        )
        excerpt = _esc(item.get("excerpt", ""))
        rows.append(
            '<div class="ev">'
            f'<div class="ev-top"><span class="cite">{_esc(item.get("label", ""))}</span>'
            f'<span class="meta">{meta}</span></div>'
            f'<p class="ev-text">“{excerpt}”</p>'
            f"{src}</div>"
        )
    body = "".join(rows) if rows else '<p class="card-sub">No filing evidence retrieved for this query.</p>'
    return (
        '<div class="card">'
        '<div class="card-head"><div>'
        '<h2 class="card-title">Filing evidence</h2>'
        '<p class="card-sub">Management disclosure, validated citations</p></div>'
        '<span class="chip filing"><span class="dot"></span>SEC filing — cited</span></div>'
        f'<div class="evidence">{body}</div></div>'
    )


def _metric_html(key: str, value: Any) -> str:
    spec = _METRIC_SPEC.get(key)
    if spec is not None:
        label, fmt, signed = spec
        try:
            text = fmt(float(value))
        except (TypeError, ValueError):
            text = _esc(value)
        cls = ""
        if signed:
            try:
                cls = " neg" if float(value) < 0 else " pos"
            except (TypeError, ValueError):
                cls = ""
        return f'<div class="metric"><div class="k">{_esc(label)}</div><div class="v{cls}">{text}</div></div>'
    label = key.replace("_", " ").title()
    return f'<div class="metric"><div class="k">{_esc(label)}</div><div class="v">{_esc(value)}</div></div>'


def _market_card_html(market: dict[str, Any]) -> str:
    metrics = market.get("metrics") or {}
    cells = "".join(_metric_html(k, v) for k, v in metrics.items())
    if not cells:
        cells = '<p class="card-sub">Market context unavailable.</p>'
    source_mode = market.get("source_mode") or "unavailable"
    is_offline = str(source_mode).lower() in {"fallback", "offline", "unavailable"}
    chip_label = "Offline snapshot" if is_offline else _esc(source_mode)
    note = (
        '<div class="snapshot-note"><b>Not live.</b> Deterministic offline market snapshot, '
        "shown as context — never treated as filing evidence.</div>"
        if is_offline
        else f'<div class="snapshot-note">Source mode: <b>{_esc(source_mode)}</b>.</div>'
    )
    return (
        '<div class="card">'
        '<div class="card-head"><div>'
        '<h2 class="card-title">Market context</h2>'
        '<p class="card-sub">Options-market-implied</p></div>'
        f'<span class="chip market"><span class="dot"></span>{chip_label}</span></div>'
        f'<div class="metrics">{cells}</div>{note}</div>'
    )


def _citation_rows_html(citations: list[dict[str, Any]]) -> str:
    rows = []
    for cite in citations:
        url = str(cite.get("source_url", "") or "")
        link = (
            f'<a class="ev-src" href="{_esc(url)}" target="_blank" rel="noopener">{_short_url(url)}</a>'
            if url
            else ""
        )
        rows.append(
            '<div class="ev">'
            f'<div class="ev-top"><span class="cite">{_esc(cite.get("label", ""))}</span>'
            f'<span class="meta"><b>{_esc(cite.get("ticker", ""))}</b> · '
            f'{_esc(cite.get("form_type", ""))} · <b>{_esc(cite.get("filing_date", ""))}</b> · '
            f'{_esc(cite.get("accession", ""))}</span></div>{link}</div>'
        )
    return "".join(rows)


def _answer_card_html(brief: dict[str, Any]) -> str:
    answer = brief.get("answer")
    if answer is None:
        gate = brief.get("answer_gate", {})
        reasons = "".join(f"<li>{_esc(r)}</li>" for r in gate.get("reasons", []))
        allowed = gate.get("allowed")
        return (
            '<div class="card">'
            '<div class="card-head"><div>'
            '<h2 class="card-title">Answer</h2>'
            '<p class="card-sub">Opt-in, grounded in cited evidence</p></div></div>'
            f'<p class="card-sub">No answer generated (gate allowed = {_esc(allowed)}). '
            "The cited evidence above is the primary output.</p>"
            f'<ul class="note-list">{reasons}</ul></div>'
        )
    citations = _citation_rows_html(answer.get("accepted_citations") or [])
    cite_block = (
        f'<p class="card-sub" style="margin-top:16px">Validated citations</p>'
        f'<div class="evidence">{citations}</div>'
        if citations
        else ""
    )
    rejected = answer.get("rejected_citations")
    rejected_block = (
        f'<div class="warn" style="margin-top:14px">Rejected (hallucinated) citations were '
        f"dropped: {_esc(rejected)}</div>"
        if rejected
        else ""
    )
    status = _esc(answer.get("status", ""))
    model = _esc(answer.get("model", ""))
    return (
        '<div class="card">'
        '<div class="card-head"><div>'
        '<h2 class="card-title">Answer</h2>'
        '<p class="card-sub">Opt-in, grounded in cited evidence</p></div></div>'
        f'<p class="answer-text">{_esc(answer.get("answer_text", ""))}</p>'
        f'<p class="subtle">status: {status} · model: {model}</p>'
        f"{cite_block}{rejected_block}</div>"
    )


def _rejected_evidence_html(rejected: list[Any]) -> str:
    if not rejected:
        return ""
    return f'<div class="warn">Rejected citations: {_esc(rejected)}</div>'


def _data_sources_html(brief: dict[str, Any]) -> str:
    rows = []
    for source in brief.get("data_sources", []):
        rows.append(
            '<div class="src-row">'
            f'<span class="lbl">{_esc(source.get("label", ""))}</span>'
            f'<span class="prov">{_esc(source.get("kind", ""))} · {_esc(source.get("provenance", ""))}</span>'
            "</div>"
        )
    notes = "".join(f"<li>{_esc(note)}</li>" for note in brief.get("notes", []))
    return (
        '<div class="card">'
        '<div class="card-head"><div>'
        '<h2 class="card-title">Data sources</h2>'
        '<p class="card-sub">What each panel is, and where it comes from</p></div></div>'
        f'{"".join(rows)}'
        f'<ul class="note-list">{notes}</ul></div>'
    )


def _markup(payload: str) -> None:
    st.markdown(payload, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pinned filing corpus...")
def _demo_service(root: str, use_voyage: bool):
    return load_demo_service(Path(root), use_voyage=use_voyage)


def main(*, hosted: bool = False) -> None:
    st.set_page_config(page_title="Filing Intelligence — SEC brief", layout="wide")

    try:
        secrets = dict(st.secrets)
    except FileNotFoundError:
        secrets = {}
    if hosted or secrets.get("APP_PASSWORD"):
        if not require_password(st, secrets):
            return
    bridge_provider_secrets(secrets)

    with st.sidebar:
        st.markdown("### Filing Intelligence")
        appearance = st.segmented_control(
            "Appearance", ["Light", "Dark"], default="Light", key="appearance"
        )
        dark = appearance == "Dark"
        inject_theme(st, dark=dark)

        with st.expander("Advanced retrieval", expanded=False):
            use_voyage = st.checkbox("Use Voyage query embeddings", value=True)
            top_k = st.number_input("Top-k", min_value=1, max_value=20, value=5, step=1)
            per_subquery_k = st.number_input("Per-subquery k", min_value=1, max_value=20, value=8, step=1)

    # inject_theme also runs in the sidebar block above; guard the (unlikely)
    # case where the segmented control returns None on first paint.
    if "appearance" not in st.session_state:
        inject_theme(st, dark=False)

    if hosted:
        try:
            service, manifest = _demo_service(str(ROOT / "artifacts/demo_slice"), use_voyage)
        except (OSError, ValueError, KeyError):
            st.error("Pinned demo data is unavailable or failed integrity checks. Contact the owner.")
            return
        st.info(
            f"Hosted demo — {len(manifest['tickers'])} of {manifest['source_company_count']} companies, "
            f"corpus pinned {manifest['built_at'][:10]}. "
            f"Latest cached filing: {manifest['latest_filing_date']}. "
            "Full corpus runs locally; this is historical disclosure, not live data. "
            f"{manifest['missing_vector_count']} chunks have no vector and use lexical scoring only."
        )
    else:
        service = build_local_api_service(root=project_root(), use_voyage=use_voyage)
    st.caption(retrieval_notice(use_voyage))
    options = company_options(service.companies()) or ["NVDA"]
    default_index = options.index("NVDA") if "NVDA" in options else 0

    # The hero renders above the controls, so bind its ticker to the selectbox's
    # persisted value (keyed "ticker") — falling back to the default on first
    # paint, or if a stale session value is no longer in the current options.
    selected_ticker = st.session_state.get("ticker", options[default_index])
    if selected_ticker not in options:
        selected_ticker = options[default_index]
    _markup(_hero_html(selected_ticker))

    ticker_col, question_col = st.columns([1, 3])
    with ticker_col:
        ticker = st.selectbox("Ticker", options, index=default_index, key="ticker")
    with question_col:
        question = st.text_area(
            "Question",
            value="What are the main business risks described in this company's filings?",
            height=90,
        )

    run_answer = st.checkbox("Generate grounded answer (opt-in, OpenAI)", value=False)
    build = st.button("Build Brief", type="primary")

    if build:
        provider = market_provider_from_metrics(DETERMINISTIC_SNAPSHOT)
        try:
            st.session_state["last_brief"] = build_unified_brief(
                service,
                question=question,
                ticker=ticker,
                top_k=int(top_k),
                per_subquery_k=int(per_subquery_k),
                market_provider=provider,
                run_answer=run_answer,
            ).to_dict()
        except Exception:
            st.session_state.pop("last_brief", None)
            st.error("The brief could not be completed. Check the question and provider availability, then retry.")
            return

    # Persist the last brief across reruns so flipping the theme (or any other
    # widget) does not wipe the results.
    brief = st.session_state.get("last_brief")
    if brief is None:
        _markup(_empty_state_html())
        return

    _markup(_answer_card_html(brief))

    evidence_column, market_column = st.columns([1.55, 1])
    with evidence_column:
        _markup(_evidence_card_html(brief["filing_evidence"]["evidence"]))
        _markup(_rejected_evidence_html(brief["filing_evidence"]["rejected_citations"]))
    with market_column:
        _markup(_market_card_html(brief["market_context"]))

    _markup(_data_sources_html(brief))


if __name__ == "__main__":
    main()
