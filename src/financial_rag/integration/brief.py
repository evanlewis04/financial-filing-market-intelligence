"""Unified analyst brief: cited filing evidence, an optional gated answer, and
options-market context, assembled with explicit data-source separation.

This composes the Step-7 market-evidence combiner with the Step-6 gated answer
path. The assembly is pure and testable; the actual OpenAI call stays opt-in and
is performed by the caller (the brief view) only when the gate allows it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from src.financial_rag.api import QueryRequest
from src.financial_rag.audit import build_evidence_quality_report
from src.financial_rag.differentiators import get_market_context
from src.financial_rag.integration.market_evidence import build_market_evidence_brief
from src.financial_rag.synthesis import check_openai_readiness, synthesize_answer_from_query_payload
from src.financial_rag.workbench import evaluate_answer_gate
from src.financial_rag.synthesis.evidence import EvidenceDeliveryError


@dataclass(frozen=True)
class UnifiedBrief:
    """Cited filing evidence, an optional gated answer, and market context."""

    question: str
    ticker: str
    answer: dict[str, Any] | None
    answer_gate: dict[str, Any]
    filing_evidence: dict[str, Any]
    market_context: dict[str, Any]
    data_sources: list[dict[str, str]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ticker": self.ticker,
            "answer": self.answer,
            "answer_gate": self.answer_gate,
            "filing_evidence": self.filing_evidence,
            "market_context": self.market_context,
            "data_sources": self.data_sources,
            "notes": list(self.notes),
        }


def assemble_unified_brief(
    query_payload: dict[str, Any],
    market_context: Any,
    *,
    question: str,
    ticker: str,
    evidence_quality_status: str,
    openai_ready: bool,
    openai_issues: list[str] | None = None,
    answer: Any | None = None,
) -> UnifiedBrief:
    """Combine retrieved evidence, market context, and an optional answer.

    Pure and offline: it never calls OpenAI. ``answer`` is whatever the caller
    chose to generate (an ``EvidenceAnswer`` or its dict), included only when the
    gate already allowed it.
    """

    base = build_market_evidence_brief(query_payload, market_context, question=question, ticker=ticker)
    gate = evaluate_answer_gate(
        query_payload,
        evidence_quality_status=evidence_quality_status,
        openai_ready=openai_ready,
        openai_issues=openai_issues,
    )
    answer_payload = answer.to_dict() if hasattr(answer, "to_dict") else answer

    notes = list(base.notes)
    if answer_payload is None:
        notes.append("No generated answer was produced; the cited evidence is the primary output.")
    else:
        notes.append("The generated answer is constrained to the cited filing evidence below; it is not market advice.")

    return UnifiedBrief(
        question=question,
        ticker=base.ticker,
        answer=answer_payload,
        answer_gate=gate.to_dict(),
        filing_evidence=base.filing_evidence,
        market_context=base.market_context,
        data_sources=base.data_sources,
        notes=notes,
    )


def build_unified_brief(
    service: Any,
    *,
    question: str,
    ticker: str,
    top_k: int = 5,
    per_subquery_k: int = 5,
    market_provider: Callable[[str], dict[str, Any]] | None = None,
    run_answer: bool = False,
    openai_client: Any | None = None,
) -> UnifiedBrief:
    """Cache-only convenience that runs retrieval and assembles the brief.

    Market context is offline-safe by default (labeled unavailable without a
    provider). The OpenAI answer is opt-in: it runs only when ``run_answer`` is
    set and the evidence/readiness gate allows it.
    """

    query_payload = service.query(
        QueryRequest(question=question, ticker=ticker, top_k=top_k, per_subquery_k=per_subquery_k)
    )
    market_context = get_market_context(ticker, provider=market_provider)
    evidence_quality = build_evidence_quality_report(query_payload)
    readiness = check_openai_readiness()
    openai_ready = readiness.status == "ready"

    gate = evaluate_answer_gate(
        query_payload,
        evidence_quality_status=evidence_quality.status,
        openai_ready=openai_ready,
        openai_issues=readiness.issues,
    )
    answer = None
    delivery_issue = ""
    if run_answer and gate.allowed:
        try:
            evidence = service.synthesis_evidence(query_payload)
            answer = synthesize_answer_from_query_payload(
                query_payload, question=question, dry_run=False, client=openai_client,
                full_evidence=evidence,
            )
        except EvidenceDeliveryError as exc:
            delivery_issue = str(exc)

    brief = assemble_unified_brief(
        query_payload,
        market_context,
        question=question,
        ticker=ticker,
        evidence_quality_status=evidence_quality.status,
        openai_ready=openai_ready,
        openai_issues=readiness.issues,
        answer=answer,
    )

    if delivery_issue:
        return replace(brief, answer_gate={"allowed": False, "reasons": [delivery_issue]},
                       notes=[*brief.notes, "Answer blocked: " + delivery_issue])
    return brief
