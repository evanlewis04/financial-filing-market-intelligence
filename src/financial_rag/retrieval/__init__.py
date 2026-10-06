"""Local retrieval primitives for filings RAG."""

from .local_dense import (
    LocalDenseRetriever,
    LocalChunkRecord,
    RetrievalFilters,
    RetrievalResult,
    cosine_similarity,
    lexical_relevance_score,
    load_local_retrieval_corpus,
)
from .packed_vectors import (
    PackedVectors,
    load_packed_vectors,
    pack_vector_cache,
    packed_vectors_exist,
)
from .rerank import (
    LexicalRerankerV1,
    RerankCandidate,
    Reranker,
    VoyageReranker,
    build_reranker,
)

__all__ = [
    "LocalDenseRetriever",
    "PackedVectors",
    "LocalChunkRecord",
    "RetrievalFilters",
    "RetrievalResult",
    "LexicalRerankerV1",
    "RerankCandidate",
    "Reranker",
    "VoyageReranker",
    "build_reranker",
    "cosine_similarity",
    "lexical_relevance_score",
    "load_local_retrieval_corpus",
    "load_packed_vectors",
    "pack_vector_cache",
    "packed_vectors_exist",
]
