"""Opt-in OpenAI Responses API synthesis over already-retrieved evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from src.financial_rag.settings import configured_secret, load_environment
from src.financial_rag.synthesis.citations import extract_citation_labels
from src.financial_rag.synthesis.evidence import (
    EvidenceDeliveryError, FullTextEvidence, validate_evidence_identity,
)


DEFAULT_OPENAI_SYNTHESIS_MODEL = "gpt-5.2"
# Chosen application limit, approved for the demo. Characters are not tokens,
# a model context-window size, or a cost guarantee. Oversize blocks explicitly.
MAX_FULL_EVIDENCE_SOURCES = 20
MAX_FULL_INPUT_CHARS = 18_000


class ResponsesClient(Protocol):
    def create_response(self, *, model: str, instructions: str, input_text: str) -> str:
        """Return generated text from a Responses API-compatible client."""


@dataclass(frozen=True)
class OpenAIReadiness:
    status: str
    api_key_configured: bool
    sdk_installed: bool
    model: str
    message: str
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceAnswer:
    status: str
    question: str
    answer_text: str
    model: str
    accepted_citations: list[dict[str, str]]
    rejected_citations: list[str]
    retrieved_labels: list[str]
    source_count: int
    dry_run: bool
    prompt_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_openai_readiness(*, model: str = DEFAULT_OPENAI_SYNTHESIS_MODEL) -> OpenAIReadiness:
    """Check local prerequisites without making an API call."""

    load_environment()
    api_key = configured_secret("OPENAI_API_KEY")
    sdk_installed = _openai_sdk_installed()
    issues: list[str] = []
    if not api_key:
        issues.append("OPENAI_API_KEY is missing or still set to a placeholder.")
    if not sdk_installed:
        issues.append("The openai Python package is not installed in this environment.")
    return OpenAIReadiness(
        status="ready" if not issues else "not_ready",
        api_key_configured=api_key is not None,
        sdk_installed=sdk_installed,
        model=model,
        message="OpenAI live answer smoke is ready." if not issues else "OpenAI live answer smoke is not ready.",
        issues=issues,
    )


def synthesize_answer_from_query_payload(
    query_payload: dict[str, Any],
    *,
    question: str,
    model: str = DEFAULT_OPENAI_SYNTHESIS_MODEL,
    dry_run: bool = True,
    client: ResponsesClient | None = None,
    full_evidence: list[FullTextEvidence] | None = None,
) -> EvidenceAnswer:
    """Build or run a citation-constrained answer over retrieved local evidence."""

    # Legacy direct callers remain explicitly excerpt-only.
    evidence = (_evidence_blocks(query_payload) if full_evidence is None
                else _full_evidence_blocks(query_payload, full_evidence))
    instructions = _instructions()
    input_text = _input_text(question=question, evidence=evidence)
    if full_evidence is not None and len(instructions) + len(input_text) > MAX_FULL_INPUT_CHARS:
        raise EvidenceDeliveryError(
            "Complete selected evidence exceeds the 18,000-character answer input budget; "
            "reduce Top-k or narrow the question. No source text was truncated."
        )
    retrieved_labels = [block["label"] for block in evidence]
    if dry_run:
        answer_text = _dry_run_answer(question=question, labels=retrieved_labels)
    else:
        active_client = client or OpenAIResponsesClient()
        answer_text = active_client.create_response(model=model, instructions=instructions, input_text=input_text)

    accepted, rejected = validate_answer_citations(answer_text, query_payload)
    status = "pass" if accepted and not rejected else "warning" if accepted else "fail"
    return EvidenceAnswer(
        status=status,
        question=question,
        answer_text=answer_text,
        model=model,
        accepted_citations=accepted,
        rejected_citations=rejected,
        retrieved_labels=retrieved_labels,
        source_count=len(evidence),
        dry_run=dry_run,
        prompt_preview=f"{instructions}\n\n{input_text}",
    )


def validate_answer_citations(answer_text: str, query_payload: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """Accept only answer citations that map to retrieved chunks."""

    labels = extract_citation_labels(answer_text)
    by_label = {
        str(result.get("citation_label", "")).upper(): result
        for result in query_payload.get("results", [])
        if result.get("citation_label")
    }
    accepted: list[dict[str, str]] = []
    rejected: list[str] = []
    for label in labels:
        result = by_label.get(label)
        if result is None:
            rejected.append(label)
            continue
        metadata = result.get("metadata", {})
        accepted.append(
            {
                "label": label,
                "ticker": str(metadata.get("ticker", "")),
                "form_type": str(metadata.get("form_type", "")),
                "filing_date": str(metadata.get("filing_date", "")),
                "accession": str(metadata.get("accession_number", "")),
                "source_url": str(result.get("source_url", "")),
                "chunk_id": str(result.get("chunk_id", "")),
            }
        )
    return accepted, rejected


class OpenAIResponsesClient:
    """Tiny adapter around the official OpenAI Python SDK."""

    def __init__(self, *, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai Python package is not installed.") from exc
        load_environment()
        resolved_key = api_key or configured_secret("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is missing or still set to a placeholder.")
        self._client = OpenAI(api_key=resolved_key)

    def create_response(self, *, model: str, instructions: str, input_text: str) -> str:
        response = self._client.responses.create(
            model=model,
            instructions=instructions,
            input=input_text,
        )
        output_text = getattr(response, "output_text", "")
        if output_text:
            return str(output_text)
        return _extract_output_text(response)


def _openai_sdk_installed() -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def _instructions() -> str:
    return (
        "You are a financial filings assistant. Answer only from the provided evidence. "
        "Every factual sentence must include at least one source label like [S1]. "
        "If the evidence is insufficient, say what is missing. Do not invent citations. "
        "Identify the reporting period for claims; filing dates are not reporting periods. "
        "Keep annual versus quarterly results and results versus outlook distinct."
    )


def _input_text(*, question: str, evidence: list[dict[str, str]]) -> str:
    lines = [f"Question: {question}", "", "Evidence:"]
    for block in evidence:
        lines.append(
            "\n".join(
                [
                    f"[{block['label']}]",
                    f"ticker: {block['ticker']}",
                    f"form_type: {block['form_type']}",
                    f"filing_date: {block['filing_date']}",
                    f"accession: {block['accession']}",
                    f"url: {block['source_url']}",
                    *([f"chunk_id: {block['chunk_id']}",
                       f"report_date: {block.get('report_date', '')}",
                       f"period_end: {block.get('period_end', '')}",
                       "evidence_mode: complete selected chunk"] if "chunk_id" in block
                      else ["evidence_mode: compact excerpt only"]),
                    f"excerpt: {block['excerpt']}",
                ]
            )
        )
    return "\n\n".join(lines)


def _full_evidence_blocks(query_payload: dict[str, Any], evidence: list[FullTextEvidence]) -> list[dict[str, str]]:
    results = query_payload.get("results", [])
    if not evidence or len(evidence) != len(results) or len(evidence) > MAX_FULL_EVIDENCE_SOURCES:
        raise EvidenceDeliveryError("Full evidence must match the selected top-k sources (maximum 20).")
    if (len({source.chunk_id for source in evidence}) != len(evidence)
            or len({source.label for source in evidence}) != len(evidence)):
        raise EvidenceDeliveryError("Full evidence contains ambiguous source identities or labels.")
    blocks = []
    for result, source in zip(results, evidence, strict=True):
        validate_evidence_identity(result, source)
        metadata = source.metadata
        blocks.append({
            "label": source.label, "chunk_id": source.chunk_id, "excerpt": source.text,
            "ticker": str(metadata.get("ticker", "")), "form_type": str(metadata.get("form_type", "")),
            "filing_date": str(metadata.get("filing_date", "")),
            "accession": str(metadata.get("accession_number", "")), "source_url": source.source_url,
            "report_date": str(metadata.get("report_date", "")), "period_end": str(metadata.get("period_end", "")),
        })
    return blocks


def _evidence_blocks(query_payload: dict[str, Any], *, max_excerpt_chars: int = 900) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for result in query_payload.get("results", []):
        metadata = result.get("metadata", {})
        label = str(result.get("citation_label", ""))
        if not label:
            continue
        excerpt = str(result.get("source_excerpt", "")).strip()
        blocks.append(
            {
                "label": label,
                "ticker": str(metadata.get("ticker", "")),
                "form_type": str(metadata.get("form_type", "")),
                "filing_date": str(metadata.get("filing_date", "")),
                "accession": str(metadata.get("accession_number", "")),
                "source_url": str(result.get("source_url", "")),
                "excerpt": excerpt[:max_excerpt_chars],
            }
        )
    return blocks


def _dry_run_answer(*, question: str, labels: list[str]) -> str:
    if not labels:
        return f"No retrieved evidence is available to answer: {question}"
    first = labels[0]
    second = labels[1] if len(labels) > 1 else labels[0]
    return (
        "Dry run only: the prompt is ready for OpenAI live testing. "
        f"The retrieved evidence should be used to answer the question with citations [{first}][{second}]."
    )


def _extract_output_text(response: Any) -> str:
    output_parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                output_parts.append(str(text))
    return "\n".join(output_parts).strip()
