"""API-shaped local service and serializers for the local filings workbench."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.financial_rag.embeddings import DEFAULT_VOYAGE_MODEL, VoyageEmbeddingProvider
from src.financial_rag.differentiators import (
    check_local_companyfacts,
    detect_filing_changes,
    get_market_context,
    summarize_language_signals,
)
from src.financial_rag.query import QueryPipeline, QueryPipelineResult, build_coverage_report
from src.financial_rag.query.coverage import CoverageReport
from src.financial_rag.query.router import KNOWN_TICKERS
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever, load_local_retrieval_corpus
from src.financial_rag.retrieval.rerank import Reranker, build_reranker
from src.financial_rag.settings import configured_secret, load_environment

if TYPE_CHECKING:
    from src.financial_rag.corpus_snapshot import CorpusSnapshot


DEFAULT_PHASE4_QUERY = "How have NVIDIA risk disclosures changed over the last year?"
MAX_TOP_K = 20


@dataclass(frozen=True)
class QueryRequest:
    question: str = DEFAULT_PHASE4_QUERY
    ticker: str = "NVDA"
    top_k: int = 5
    per_subquery_k: int = 5
    # Point-in-time query (Phase 1 Stage 2): when set, retrieval is hard-filtered
    # to filings knowable at this instant (filed_at <= as_of). None = full corpus.
    as_of: datetime | None = None


class LocalApiError(ValueError):
    """Structured local API error used by scripts and optional FastAPI routes."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ConstantQueryEmbedder:
    """Offline smoke embedder that avoids provider calls."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = max(dimensions, 1)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * (self.dimensions - 1))] for _ in texts]


class LocalRagApiService:
    """Small endpoint-shaped facade over the Phase 3 query pipeline."""

    def __init__(
        self,
        *,
        chunks: list[LocalChunkRecord],
        retriever: LocalDenseRetriever,
        root: Path | str = Path("."),
        reranker: Reranker | None = None,
        supported_tickers: set[str] | None = None,
    ) -> None:
        self.supported_tickers = set(KNOWN_TICKERS if supported_tickers is None else supported_tickers)
        self.chunks = chunks
        self.retriever = retriever
        self.root = Path(root)
        self.reranker = reranker
        self.pipeline = QueryPipeline(retriever=retriever, chunks=chunks, reranker=reranker)

    def synthesis_evidence(self, query_payload: dict[str, Any]):
        """Resolve only selected IDs against this service's already-loaded corpus."""
        from src.financial_rag.synthesis.evidence import resolve_full_text_evidence

        return resolve_full_text_evidence(query_payload, self.chunks)

    def health(self) -> dict[str, Any]:
        status = "ok" if self.chunks and self.retriever.embeddings else "empty_cache"
        return {
            "status": status,
            "service": "financial_rag_local_api",
            "chunk_count": len(self.chunks),
            "embedding_count": len(self.retriever.embeddings),
        }

    def coverage(self, *, tickers: list[str] | None = None) -> dict[str, Any]:
        if tickers:
            for ticker in tickers:
                _validate_ticker(ticker, supported_tickers=self.supported_tickers)
        return serialize_coverage_report(build_coverage_report(self.chunks, tickers=tickers))

    def query(self, request: QueryRequest) -> dict[str, Any]:
        _validate_request(request, supported_tickers=self.supported_tickers)
        self._require_cache(require_embeddings=False)
        result = self.pipeline.run(
            request.question,
            default_ticker=request.ticker,
            top_k=request.top_k,
            per_subquery_k=request.per_subquery_k,
            as_of=request.as_of,
        )
        payload = serialize_query_result(result)
        if not payload["results"]:
            raise LocalApiError(
                status_code=404,
                code="empty_retrieval_results",
                message="No local chunks matched the query and filters.",
                details={"ticker": request.ticker.upper(), "question": request.question},
            )
        result_chunks = _chunks_by_id(self.chunks, [item["chunk_id"] for item in payload["results"]])
        payload["language_signals"] = [
            summary.to_dict() for summary in summarize_language_signals(result_chunks, group_by="document_id")
        ]
        payload["market_context"] = get_market_context(request.ticker).to_dict()
        return payload

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return serialize_chunk(chunk)
        return None

    def require_chunk(self, chunk_id: str) -> dict[str, Any]:
        payload = self.get_chunk(chunk_id)
        if payload is None:
            raise LocalApiError(
                status_code=404,
                code="chunk_not_found",
                message="No local chunk matched the requested chunk id.",
                details={"chunk_id": chunk_id},
            )
        return payload

    def differentiators(self, *, ticker: str, fact_name: str = "Revenues") -> dict[str, Any]:
        _validate_ticker(ticker, supported_tickers=self.supported_tickers)
        self._require_cache(require_embeddings=False)
        ticker_chunks = [
            chunk for chunk in self.chunks if str(chunk.metadata.get("ticker", "")).upper() == ticker.upper()
        ]
        facts_path = self.root / "data" / "companyfacts" / f"{ticker.upper()}.json"
        return {
            "ticker": ticker.upper(),
            "changes": [record.to_dict() for record in detect_filing_changes(self.chunks, ticker=ticker)],
            "language_signals": [
                summary.to_dict() for summary in summarize_language_signals(ticker_chunks, group_by="item_number")
            ],
            "xbrl": check_local_companyfacts(
                ticker=ticker,
                fact_name=fact_name,
                facts_path=facts_path,
            ).to_dict(),
            "market_context": get_market_context(ticker).to_dict(),
        }

    def companies(self) -> dict[str, Any]:
        """List the locally cached, queryable companies with coverage summary."""

        self._require_cache(require_embeddings=False)
        report = build_coverage_report(self.chunks)
        document_counts = self._document_counts()
        companies = [
            {
                "ticker": ticker,
                "chunk_count": coverage.chunk_count,
                "document_count": document_counts.get(ticker, 0),
                "form_types": coverage.form_types,
                "ex99_chunk_count": coverage.ex99_chunk_count,
                "has_press_release": coverage.has_press_release,
                "has_cfo_commentary": coverage.has_cfo_commentary,
                "has_prepared_remarks": coverage.has_prepared_remarks,
                "gaps": coverage.gaps,
            }
            for ticker, coverage in report.tickers.items()
        ]
        return {"company_count": len(companies), "companies": companies}

    def documents(self, *, ticker: str | None = None) -> dict[str, Any]:
        """List cached filing documents, optionally filtered by ticker."""

        self._require_cache(require_embeddings=False)
        wanted = _validate_ticker(ticker, supported_tickers=self.supported_tickers) if ticker else None
        documents: dict[str, dict[str, Any]] = {}
        for chunk in self.chunks:
            metadata = chunk.metadata
            chunk_ticker = str(metadata.get("ticker", "")).upper()
            if wanted is not None and chunk_ticker != wanted:
                continue
            document_id = str(metadata.get("document_id", ""))
            if not document_id:
                continue
            entry = documents.get(document_id)
            if entry is None:
                documents[document_id] = {
                    "document_id": document_id,
                    "ticker": chunk_ticker,
                    "form_type": str(metadata.get("form_type", "")),
                    "accession_number": str(metadata.get("accession_number", "")),
                    "filing_date": str(metadata.get("filing_date", "")),
                    "document_role": str(metadata.get("document_role", "")),
                    "exhibit_type": str(metadata.get("exhibit_type", "")),
                    "source_url": str(metadata.get("source_url", "")),
                    "chunk_count": 1,
                }
            else:
                entry["chunk_count"] += 1
        ordered = sorted(documents.values(), key=lambda item: (item["ticker"], item["filing_date"], item["document_id"]))
        return {"document_count": len(ordered), "documents": ordered}

    def market_context(self, *, ticker: str) -> dict[str, Any]:
        """Return market-data context provenance for one queryable ticker."""

        normalized = _validate_ticker(ticker, supported_tickers=self.supported_tickers)
        return {"ticker": normalized, "market_context": get_market_context(normalized).to_dict()}

    def _document_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        seen: set[str] = set()
        for chunk in self.chunks:
            document_id = str(chunk.metadata.get("document_id", ""))
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            ticker = str(chunk.metadata.get("ticker", "")).upper()
            counts[ticker] = counts.get(ticker, 0) + 1
        return counts

    def _require_cache(self, *, require_embeddings: bool) -> None:
        if not self.chunks:
            raise LocalApiError(
                status_code=503,
                code="missing_local_chunks",
                message="No local chunks are cached under data/filings/chunks.",
            )
        if require_embeddings and not self.retriever.embeddings:
            raise LocalApiError(
                status_code=503,
                code="missing_cached_embeddings",
                message="No local vector-cache embeddings are available under data/vector_cache.",
            )


def build_local_api_service(
    *,
    root: Path | str,
    use_voyage: bool = True,
    reranker: str = "none",
    snapshot: "CorpusSnapshot | None" = None,
    supported_tickers: set[str] | None = None,
) -> LocalRagApiService:
    """Build the local service from cached chunks and vectors.

    ``reranker`` selects the rerank stage: "lexical" (deterministic, offline,
    default), "none" (dense+lexical baseline), or "voyage" (opt-in online).
    ``snapshot`` pins the corpus to a recorded composition (Phase 1 Stage 3);
    ``None`` loads the full live corpus (no behavior change).
    """

    load_environment()
    chunks, embeddings = load_local_retrieval_corpus(root=root, snapshot=snapshot)
    if use_voyage:
        api_key = configured_secret("VOYAGE_API_KEY")
        if api_key:
            embedder = VoyageEmbeddingProvider(api_key=api_key, model=DEFAULT_VOYAGE_MODEL)
        else:
            embedder = ConstantQueryEmbedder(1)
    else:
        embedder = ConstantQueryEmbedder(1)
    retriever = LocalDenseRetriever(chunks=chunks, embeddings=embeddings, query_embedder=embedder)
    reranker_impl = build_reranker(reranker, chunk_texts=[chunk.chunk_text for chunk in chunks])
    return LocalRagApiService(chunks=chunks, retriever=retriever, root=root, reranker=reranker_impl,
                              supported_tickers=supported_tickers)


def create_fastapi_app(service: LocalRagApiService) -> Any:
    """Create a FastAPI app when FastAPI is installed.

    The project tests use `LocalRagApiService` directly so Phase 4 does not add
    a hard FastAPI dependency.
    """

    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FastAPI is not installed; use LocalRagApiService directly.") from exc

    app = FastAPI(title="Financial RAG Local Workbench")

    @app.exception_handler(LocalApiError)
    def local_api_error_handler(_request: Any, exc: LocalApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/companies")
    def companies() -> dict[str, Any]:
        return service.companies()

    @app.get("/coverage")
    def coverage(ticker: str | None = None) -> dict[str, Any]:
        tickers = [ticker] if ticker else None
        return service.coverage(tickers=tickers)

    @app.get("/documents")
    def documents(ticker: str | None = None) -> dict[str, Any]:
        return service.documents(ticker=ticker)

    @app.post("/query")
    def query(request: dict[str, Any]) -> dict[str, Any]:
        return service.query(_query_request_from_payload(request))

    @app.get("/chunks/{chunk_id}")
    def chunk(chunk_id: str) -> dict[str, Any]:
        return service.require_chunk(chunk_id)

    @app.get("/differentiators/{ticker}")
    def differentiators(ticker: str, fact_name: str = "Revenues") -> dict[str, Any]:
        return service.differentiators(ticker=ticker, fact_name=fact_name)

    @app.get("/market-context/{ticker}")
    def market_context(ticker: str) -> dict[str, Any]:
        return service.market_context(ticker=ticker)

    return app


def call_local_api_endpoint(
    service: LocalRagApiService,
    *,
    method: str,
    path: str,
    query_params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise endpoint-shaped contracts without requiring FastAPI."""

    params = query_params or {}
    try:
        payload = _dispatch_local_api_endpoint(
            service,
            method=method.upper(),
            path=path,
            query_params=params,
            body=body or {},
        )
    except LocalApiError as exc:
        return {"status_code": exc.status_code, "payload": exc.to_dict()}
    return {"status_code": 200, "payload": payload}


def api_endpoint_manifest() -> list[dict[str, str]]:
    return [
        {"method": "GET", "path": "/health", "description": "Local cache and service status."},
        {"method": "GET", "path": "/companies", "description": "Cached queryable companies and coverage summary."},
        {"method": "GET", "path": "/coverage", "description": "Cached ticker/form/source coverage."},
        {"method": "GET", "path": "/documents", "description": "Cached filing documents, optionally filtered by ticker."},
        {"method": "POST", "path": "/query", "description": "Retrieve evidence chunks for a question."},
        {"method": "GET", "path": "/chunks/{chunk_id}", "description": "Fetch one local chunk by id."},
        {"method": "GET", "path": "/differentiators/{ticker}", "description": "Local differentiator payload."},
        {"method": "GET", "path": "/market-context/{ticker}", "description": "Market-data context provenance for one ticker."},
    ]


def serialize_query_result(result: QueryPipelineResult) -> dict[str, Any]:
    return {
        "routed_query": {
            "question": result.routed_query.question,
            "query_type": result.routed_query.query_type.value,
            "filters": asdict(result.routed_query.filters),
            "trace": result.routed_query.trace,
        },
        "subqueries": [
            {
                "subquery_id": subquery.subquery_id,
                "query": subquery.query,
                "filters": asdict(subquery.filters),
                "trace": subquery.trace,
            }
            for subquery in result.subqueries
        ],
        "results": [
            {
                "chunk_id": item.chunk_id,
                "rank": item.rank,
                "score": item.dense_score,
                "citation_label": item.citation_label,
                "source_url": item.source_url,
                "source_excerpt": item.source_excerpt,
                "metadata": item.metadata,
                "subquery_id": item.subquery_id,
                "trace": item.trace,
                "parent_context": asdict(item.parent_context) if item.parent_context else None,
            }
            for item in result.results
        ],
        "citations": {
            "accepted": [asdict(citation) for citation in result.citation_validation.accepted],
            "rejected": result.citation_validation.rejected,
        },
        "coverage": serialize_coverage_report(result.coverage),
        "trace": result.trace,
    }


def serialize_coverage_report(report: CoverageReport) -> dict[str, Any]:
    return {"tickers": {ticker: asdict(coverage) for ticker, coverage in report.tickers.items()}}


def serialize_chunk(chunk: LocalChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_text": chunk.chunk_text,
        "metadata": chunk.metadata,
    }


def _embedding_dimensions(embeddings: Mapping[str, list[float]]) -> int:
    for vector in embeddings.values():
        return len(vector)
    return 1


def _chunks_by_id(chunks: list[LocalChunkRecord], chunk_ids: list[str]) -> list[LocalChunkRecord]:
    wanted = set(chunk_ids)
    return [chunk for chunk in chunks if chunk.chunk_id in wanted]


def _dispatch_local_api_endpoint(
    service: LocalRagApiService,
    *,
    method: str,
    path: str,
    query_params: dict[str, Any],
    body: dict[str, Any],
) -> dict[str, Any]:
    normalized_path = path.rstrip("/") or "/"
    if method == "GET" and normalized_path == "/health":
        return service.health()
    if method == "GET" and normalized_path == "/companies":
        return service.companies()
    if method == "GET" and normalized_path == "/coverage":
        ticker = str(query_params.get("ticker", "")).strip()
        return service.coverage(tickers=[ticker] if ticker else None)
    if method == "GET" and normalized_path == "/documents":
        ticker = str(query_params.get("ticker", "")).strip()
        return service.documents(ticker=ticker or None)
    if method == "POST" and normalized_path == "/query":
        return service.query(_query_request_from_payload(body))
    if method == "GET" and normalized_path.startswith("/chunks/"):
        return service.require_chunk(normalized_path.removeprefix("/chunks/"))
    if method == "GET" and normalized_path.startswith("/differentiators/"):
        ticker = normalized_path.removeprefix("/differentiators/")
        fact_name = str(query_params.get("fact_name", "Revenues"))
        return service.differentiators(ticker=ticker, fact_name=fact_name)
    if method == "GET" and normalized_path.startswith("/market-context/"):
        return service.market_context(ticker=normalized_path.removeprefix("/market-context/"))
    raise LocalApiError(
        status_code=404,
        code="endpoint_not_found",
        message="No local API endpoint matched the requested method and path.",
        details={"method": method, "path": path},
    )


def _query_request_from_payload(payload: dict[str, Any]) -> QueryRequest:
    return QueryRequest(
        question=str(payload.get("question", DEFAULT_PHASE4_QUERY)),
        ticker=str(payload.get("ticker", "NVDA")),
        top_k=int(payload.get("top_k", 5)),
        per_subquery_k=int(payload.get("per_subquery_k", 5)),
        as_of=_parse_as_of(payload.get("as_of")),
    )


def _parse_as_of(value: Any) -> datetime | None:
    """Parse an optional ISO-8601 date or datetime from the request payload.

    Accepts ``2026-03-01`` or a full ISO datetime (with ``Z`` or an offset).
    A malformed value is a client error, not a silent None, so an as-of query is
    never quietly run against the full corpus.
    """

    if value in (None, ""):
        return None
    text = str(value).strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise LocalApiError(
            status_code=400,
            code="invalid_as_of",
            message="as_of must be an ISO-8601 date or datetime (e.g. 2026-03-01).",
            details={"as_of": text},
        ) from exc


def _validate_request(request: QueryRequest, *, supported_tickers: set[str] | None = None) -> None:
    if not request.question.strip():
        raise LocalApiError(
            status_code=400,
            code="invalid_question",
            message="Query question must be a non-empty string.",
        )
    _validate_ticker(request.ticker, supported_tickers=supported_tickers)
    _validate_k("top_k", request.top_k)
    _validate_k("per_subquery_k", request.per_subquery_k)


def _validate_ticker(ticker: str, *, supported_tickers: set[str] | None = None) -> str:
    universe = KNOWN_TICKERS if supported_tickers is None else supported_tickers
    normalized = ticker.strip().upper()
    if not normalized:
        raise LocalApiError(status_code=400, code="invalid_ticker", message="Ticker must be non-empty.")
    if normalized not in universe:
        raise LocalApiError(
            status_code=400,
            code="unsupported_ticker",
            message="Ticker is outside the configured financial RAG universe.",
            details={"ticker": normalized, "supported_tickers": sorted(universe)},
        )
    return normalized


def _validate_k(name: str, value: int) -> None:
    if value < 1 or value > MAX_TOP_K:
        raise LocalApiError(
            status_code=400,
            code=f"invalid_{name}",
            message=f"{name} must be between 1 and {MAX_TOP_K}.",
            details={name: value},
        )
