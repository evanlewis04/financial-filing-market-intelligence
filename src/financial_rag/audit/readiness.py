"""Deterministic local-cache readiness checks for recruiter demos."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.financial_rag.query.router import KNOWN_TICKERS
from src.financial_rag.retrieval import LocalChunkRecord


PRIMARY_FORMS = {"10-K", "10-Q"}
EXPECTED_EX99_TYPES = {"PRESS_RELEASE", "CFO_COMMENTARY", "PREPARED_REMARKS"}


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    severity: str
    message: str
    ticker: str = ""
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    chunk_count: int
    embedding_count: int
    ticker_count: int
    unsupported_tickers: list[str]
    missing_embedding_count: int
    missing_item_metadata_count: int
    ex99_coverage: dict[str, dict[str, Any]]
    companyfacts_available: dict[str, bool]
    issues: list[ReadinessIssue]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


def build_readiness_report(
    chunks: list[LocalChunkRecord],
    embeddings: Mapping[str, list[float]],
    *,
    tickers: list[str] | None = None,
    root: Path | str = Path("."),
    supported_tickers: set[str] | None = None,
) -> ReadinessReport:
    """Inspect local chunks/vectors without fetching external data."""

    root_path = Path(root)
    supported = supported_tickers or set(KNOWN_TICKERS)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    ticker_set = _ticker_set(chunks, tickers)
    unsupported = sorted(ticker for ticker in ticker_set if ticker not in supported)
    missing_embedding_ids = sorted(chunk_id for chunk_id in chunk_ids if chunk_id not in embeddings)
    missing_item_chunks = [
        chunk.chunk_id
        for chunk in chunks
        if _form_type(chunk) in PRIMARY_FORMS and not str(chunk.metadata.get("item_number", "")).strip()
    ]
    ex99_coverage = _ex99_coverage(chunks, ticker_set)
    companyfacts_available = {
        ticker: (root_path / "data" / "companyfacts" / f"{ticker}.json").exists() for ticker in sorted(ticker_set)
    }

    issues: list[ReadinessIssue] = []
    if unsupported:
        issues.append(
            ReadinessIssue(
                code="unsupported_ticker",
                severity="warning",
                message="Ticker is outside the initial financial RAG universe.",
                count=len(unsupported),
                examples=unsupported[:5],
            )
        )
    if missing_embedding_ids:
        issues.append(
            ReadinessIssue(
                code="missing_embeddings",
                severity="fail",
                message="Some cached chunks do not have vector-cache embeddings.",
                count=len(missing_embedding_ids),
                examples=missing_embedding_ids[:5],
            )
        )
    if missing_item_chunks:
        issues.append(
            ReadinessIssue(
                code="missing_item_metadata",
                severity="warning",
                message="Some 10-K/10-Q chunks are missing SEC item metadata.",
                count=len(missing_item_chunks),
                examples=missing_item_chunks[:5],
            )
        )
    for ticker, coverage in ex99_coverage.items():
        missing_types = coverage["missing_expected_types"]
        if not coverage["chunk_count"]:
            issues.append(
                ReadinessIssue(
                    code="missing_ex99_coverage",
                    severity="warning",
                    message="No EX-99 chunks are available for this ticker.",
                    ticker=ticker,
                )
            )
        elif missing_types:
            issues.append(
                ReadinessIssue(
                    code="sparse_ex99_coverage",
                    severity="info",
                    message="EX-99 coverage is present but missing some expected exhibit categories.",
                    ticker=ticker,
                    count=len(missing_types),
                    examples=missing_types,
                )
            )
    for ticker, available in companyfacts_available.items():
        if not available:
            issues.append(
                ReadinessIssue(
                    code="missing_companyfacts",
                    severity="info",
                    message="Local SEC companyfacts JSON is not available for XBRL scaffolding.",
                    ticker=ticker,
                )
            )

    status = "fail" if any(issue.severity == "fail" for issue in issues) else "warning" if issues else "ready"
    return ReadinessReport(
        status=status,
        chunk_count=len(chunks),
        embedding_count=len(embeddings),
        ticker_count=len(ticker_set),
        unsupported_tickers=unsupported,
        missing_embedding_count=len(missing_embedding_ids),
        missing_item_metadata_count=len(missing_item_chunks),
        ex99_coverage=ex99_coverage,
        companyfacts_available=companyfacts_available,
        issues=issues,
    )


def write_json_report(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _ticker_set(chunks: list[LocalChunkRecord], tickers: list[str] | None) -> set[str]:
    found = {str(chunk.metadata.get("ticker", "")).strip().upper() for chunk in chunks}
    found.discard("")
    if tickers:
        found.update(ticker.strip().upper() for ticker in tickers if ticker.strip())
    return found


def _form_type(chunk: LocalChunkRecord) -> str:
    return str(chunk.metadata.get("form_type", "")).strip().upper()


def _ex99_coverage(chunks: list[LocalChunkRecord], tickers: set[str]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for ticker in sorted(tickers):
        ticker_chunks = [
            chunk
            for chunk in chunks
            if str(chunk.metadata.get("ticker", "")).strip().upper() == ticker and _is_ex99(chunk)
        ]
        exhibit_types = sorted(
            {
                str(chunk.metadata.get("exhibit_type", "")).strip().upper()
                for chunk in ticker_chunks
                if str(chunk.metadata.get("exhibit_type", "")).strip()
            }
        )
        coverage[ticker] = {
            "chunk_count": len(ticker_chunks),
            "exhibit_types": exhibit_types,
            "missing_expected_types": sorted(EXPECTED_EX99_TYPES - set(exhibit_types)),
        }
    return coverage


def _is_ex99(chunk: LocalChunkRecord) -> bool:
    form_type = _form_type(chunk)
    document_role = str(chunk.metadata.get("document_role", "")).strip().lower()
    exhibit_type = str(chunk.metadata.get("exhibit_type", "")).strip().upper()
    return form_type == "EX-99" or document_role == "exhibit" or exhibit_type.startswith("EX-99")
