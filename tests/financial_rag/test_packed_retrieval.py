"""Packed vector store, vectorized dense scoring, and the per-result dense score."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.financial_rag.retrieval import (
    LocalChunkRecord,
    LocalDenseRetriever,
    PackedVectors,
    RetrievalFilters,
    cosine_similarity,
    load_local_retrieval_corpus,
    load_packed_vectors,
    pack_vector_cache,
    packed_vectors_exist,
)
from src.financial_rag.storage import LocalRagStore


def test_each_result_reports_its_own_dense_score() -> None:
    """Regression: dense_score used to leak the last scanned chunk's value.

    The reported score came from the loop variable left over after scoring, so
    every result echoed the final chunk considered instead of its own.
    """

    retriever = LocalDenseRetriever(
        chunks=[
            _chunk("aligned", "risk factors describe export controls"),
            _chunk("orthogonal", "risk factors describe export controls"),
        ],
        embeddings={"aligned": [1.0, 0.0], "orthogonal": [0.0, 1.0]},
    )

    results = retriever.search(query_vector=[1.0, 0.0], top_k=2)

    by_id = {result.chunk_id: result.dense_score for result in results}
    assert by_id["aligned"] == 1.0
    assert by_id["orthogonal"] == 0.0


def test_missing_vector_scores_zero_without_dropping_the_chunk() -> None:
    retriever = LocalDenseRetriever(
        chunks=[_chunk("has_vector", "export controls"), _chunk("no_vector", "export controls")],
        embeddings={"has_vector": [1.0, 0.0]},
    )

    results = retriever.search(query_vector=[1.0, 0.0], top_k=2)

    by_id = {result.chunk_id: result.dense_score for result in results}
    assert by_id == {"has_vector": 1.0, "no_vector": 0.0}


def test_pack_and_load_round_trip_behaves_like_the_dict_it_replaces(tmp_path: Path) -> None:
    cache_dir = tmp_path / "vector_cache"
    vectors = {"a": [1.0, 0.0, 0.5], "b": [0.25, -0.5, 1.0]}
    _write_vector_cache(cache_dir, vectors)

    meta = pack_vector_cache(vector_cache_dir=cache_dir, out_dir=tmp_path / "packed")

    assert meta["count"] == 2
    assert meta["dimensions"] == 3
    assert meta["dtype"] == "float16"
    assert packed_vectors_exist(tmp_path / "packed")

    packed = load_packed_vectors(tmp_path / "packed")

    assert len(packed) == 2
    assert set(packed) == {"a", "b"}
    assert "a" in packed and "zz" not in packed
    assert packed.get("zz") is None
    assert packed.dimensions == 3
    assert packed.row_for("b") is not None
    assert packed.row_for("zz") is None
    # float16 storage: values match the source cache within quantization error.
    for chunk_id, expected in vectors.items():
        assert packed[chunk_id] == [float(np.float16(value)) for value in expected]
    assert len(list(packed.values())) == 2


def test_membership_does_not_materialize_rows() -> None:
    """The readiness audit probes membership once per chunk (readiness.py:66).

    Mapping's default __contains__ routes through __getitem__, which would turn
    each probe into a 1024-float list build on the real corpus.
    """

    calls: list[str] = []

    class CountingPackedVectors(PackedVectors):
        def __getitem__(self, chunk_id: str) -> list[float]:
            calls.append(chunk_id)
            return super().__getitem__(chunk_id)

    packed = CountingPackedVectors(
        chunk_ids=["a", "b"],
        matrix=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
    )

    assert "a" in packed
    assert "missing" not in packed
    assert calls == []

    # The row is still reachable the expensive way when a caller wants it.
    assert packed["a"] == [1.0, 0.0]
    assert calls == ["a"]


def test_pack_respects_a_chunk_id_allowlist(tmp_path: Path) -> None:
    cache_dir = tmp_path / "vector_cache"
    _write_vector_cache(cache_dir, {"keep": [1.0, 0.0], "drop": [0.0, 1.0]})

    meta = pack_vector_cache(
        vector_cache_dir=cache_dir,
        out_dir=tmp_path / "packed",
        chunk_ids=["keep"],
    )

    packed = load_packed_vectors(tmp_path / "packed")
    assert meta["count"] == 1
    assert meta["restricted_to_chunk_ids"] is True
    assert set(packed) == {"keep"}


def test_pack_rejects_mixed_dimensions_without_publishing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "vector_cache"
    _write_vector_cache(cache_dir, {"a_ok": [1.0, 0.0], "b_odd": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="Mixed vector dimensions"):
        pack_vector_cache(vector_cache_dir=cache_dir, out_dir=tmp_path / "packed")
    assert not packed_vectors_exist(tmp_path / "packed")


def test_packed_search_matches_the_per_chunk_cosine_path() -> None:
    """The vectorized matmul must reproduce the pure-Python ranking and scores."""

    rng = np.random.default_rng(7)
    raw = {f"chunk-{index}": rng.normal(size=32).tolist() for index in range(25)}
    chunks = [_chunk(chunk_id, f"export controls demand text {chunk_id}") for chunk_id in raw]
    query = rng.normal(size=32).tolist()

    dict_results = LocalDenseRetriever(chunks=chunks, embeddings=raw).search(
        query_vector=query, top_k=10
    )
    packed_results = LocalDenseRetriever(chunks=chunks, embeddings=_packed(raw)).search(
        query_vector=query, top_k=10
    )

    assert [result.chunk_id for result in packed_results] == [
        result.chunk_id for result in dict_results
    ]
    for packed_result, dict_result in zip(packed_results, dict_results, strict=True):
        assert math.isclose(packed_result.dense_score, dict_result.dense_score, abs_tol=5e-3)
        assert math.isclose(packed_result.score, dict_result.score, abs_tol=2e-3)


def test_packed_dense_score_matches_cosine_similarity_directly() -> None:
    raw = {"a": [0.6, 0.8, 0.0], "b": [-1.0, 0.25, 0.5]}
    chunks = [_chunk("a", "export controls"), _chunk("b", "export controls")]
    query = [0.2, 0.9, -0.3]

    results = LocalDenseRetriever(chunks=chunks, embeddings=_packed(raw)).search(
        query_vector=query, top_k=2
    )

    for result in results:
        assert math.isclose(
            result.dense_score, cosine_similarity(query, raw[result.chunk_id]), abs_tol=5e-3
        )


def test_packed_path_scores_zero_on_a_dimension_mismatch() -> None:
    """Mirrors cosine_similarity: the offline evals query with a 1-dim vector."""

    chunks = [_chunk("a", "export controls"), _chunk("b", "export controls")]
    packed = _packed({"a": [1.0, 0.0], "b": [0.0, 1.0]})

    results = LocalDenseRetriever(chunks=chunks, embeddings=packed).search(
        query_vector=[1.0], top_k=2
    )

    assert [result.dense_score for result in results] == [0.0, 0.0]


def test_packed_path_keeps_filters_and_unpacked_chunks() -> None:
    chunks = [
        _chunk("nvda", "export controls", ticker="NVDA"),
        _chunk("amd", "export controls", ticker="AMD"),
        _chunk("nvda_unpacked", "export controls", ticker="NVDA"),
    ]
    packed = _packed({"nvda": [1.0, 0.0], "amd": [1.0, 0.0]})

    results = LocalDenseRetriever(chunks=chunks, embeddings=packed).search(
        query_vector=[1.0, 0.0],
        filters=RetrievalFilters(ticker="NVDA"),
        top_k=5,
    )

    by_id = {result.chunk_id: result.dense_score for result in results}
    assert set(by_id) == {"nvda", "nvda_unpacked"}
    assert by_id["nvda"] == 1.0
    assert by_id["nvda_unpacked"] == 0.0


def test_corpus_loader_prefers_the_packed_artifact(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    _write_chunk_jsonl(store, "a")
    _write_vector_cache(store.vector_cache_dir, {"a": [1.0, 0.0]})

    _chunks, fallback = load_local_retrieval_corpus(root=tmp_path)
    assert not isinstance(fallback, PackedVectors)
    assert fallback["a"] == [1.0, 0.0]

    pack_vector_cache(vector_cache_dir=store.vector_cache_dir, out_dir=store.packed_dir)
    _chunks, packed = load_local_retrieval_corpus(root=tmp_path)

    assert isinstance(packed, PackedVectors)
    assert packed["a"] == [1.0, 0.0]


def _packed(vectors: dict[str, list[float]]) -> PackedVectors:
    chunk_ids = list(vectors)
    matrix = np.asarray([vectors[chunk_id] for chunk_id in chunk_ids], dtype=np.float16)
    return PackedVectors(chunk_ids=chunk_ids, matrix=matrix)


def _write_vector_cache(cache_dir: Path, vectors: dict[str, list[float]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for chunk_id, vector in vectors.items():
        (cache_dir / f"{chunk_id}.json").write_text(
            json.dumps({"chunk_id": chunk_id, "embedding": vector}), encoding="utf-8"
        )


def _write_chunk_jsonl(store: LocalRagStore, chunk_id: str) -> None:
    row: dict[str, Any] = {
        "chunk_id": chunk_id,
        "chunk_text": "Export controls may affect demand.",
        "metadata": _chunk(chunk_id, "").metadata,
    }
    store.chunks_path("doc").write_text(json.dumps(row) + "\n", encoding="utf-8")


def _chunk(chunk_id: str, text: str, *, ticker: str = "NVDA") -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc",
            "ticker": ticker,
            "form_type": "10-K",
            "filing_date": "2026-02-25",
            "accession_number": "0001045810-26-000021",
            "source_url": "https://www.sec.gov/Archives/doc.htm",
            "document_role": "primary",
            "exhibit_type": "",
            "item_number": "1A",
        },
    )


def test_duplicate_cache_ids_keep_last_sorted_file(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, vector in [("a", [1.0, 0.0]), ("z", [0.0, 1.0])]:
        (cache / f"{name}.json").write_text(json.dumps({"chunk_id": "same", "embedding": vector}))
    meta = pack_vector_cache(vector_cache_dir=cache, out_dir=tmp_path / "packed")
    assert meta["duplicate_chunk_ids"] == 1
    assert load_packed_vectors(tmp_path / "packed")["same"] == [0.0, 1.0]


def test_existing_pack_is_not_overwritten(tmp_path: Path) -> None:
    cache, out = tmp_path / "cache", tmp_path / "packed"
    _write_vector_cache(cache, {"a": [1.0, 0.0]})
    pack_vector_cache(vector_cache_dir=cache, out_dir=out)
    original = (out / "vectors.npy").read_bytes()
    with pytest.raises(FileExistsError):
        pack_vector_cache(vector_cache_dir=cache, out_dir=out)
    assert (out / "vectors.npy").read_bytes() == original


@pytest.mark.parametrize("vector", [[float("nan"), 0.0], [float("inf"), 0.0], [], [1e10, 0.0]])
def test_invalid_vectors_do_not_publish(tmp_path: Path, vector: list[float]) -> None:
    cache = tmp_path / "cache"
    _write_vector_cache(cache, {"bad": vector})
    with pytest.raises(ValueError):
        pack_vector_cache(vector_cache_dir=cache, out_dir=tmp_path / "packed")
    assert not packed_vectors_exist(tmp_path / "packed")


def test_partial_pack_is_not_selected(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    _write_vector_cache(store.vector_cache_dir, {"a": [1.0, 0.0]})
    np.save(store.packed_dir / "vectors.npy", np.zeros((1, 2), dtype=np.float16))
    (store.packed_dir / "chunk_ids.json").write_text('["a"]')
    _, embeddings = load_local_retrieval_corpus(root=tmp_path)
    assert isinstance(embeddings, dict)
    assert embeddings["a"] == [1.0, 0.0]


def test_loader_rejects_metadata_mismatch(tmp_path: Path) -> None:
    cache, out = tmp_path / "cache", tmp_path / "packed"
    _write_vector_cache(cache, {"a": [1.0, 0.0]})
    pack_vector_cache(vector_cache_dir=cache, out_dir=out)
    (out / "meta.json").write_text('{"count": 2, "dimensions": 2, "dtype": "float16"}')
    with pytest.raises(ValueError, match="metadata does not match"):
        load_packed_vectors(out)


def test_zero_vectors_and_stable_ties_match_dict() -> None:
    raw = {"zero": [0.0, 0.0], "a": [1.0, 0.0], "b": [1.0, 0.0]}
    chunks = [_chunk(k, "same text") for k in raw] + [_chunk("missing", "same text")]
    for query in ([1.0, 0.0], [0.0, 0.0], []):
        old = LocalDenseRetriever(chunks=chunks, embeddings=raw).search(query_vector=query, top_k=8)
        new = LocalDenseRetriever(chunks=chunks, embeddings=_packed(raw)).search(query_vector=query, top_k=8)
        assert [(r.chunk_id, r.score, r.dense_score) for r in new] == [(r.chunk_id, r.score, r.dense_score) for r in old]


def test_float16_can_collapse_a_near_tie() -> None:
    # Quantization removes a real but tiny difference. Stable sort then retains
    # corpus order. Do not claim universal dense ranking equivalence.
    raw = {"first": [1.0, 1.0], "second": [1.0001, 1.0]}
    chunks = [_chunk(k, "same text") for k in raw]
    old = LocalDenseRetriever(chunks=chunks, embeddings=raw).search(query_vector=[1.0, 0.0])
    new = LocalDenseRetriever(chunks=chunks, embeddings=_packed(raw)).search(query_vector=[1.0, 0.0])
    assert [r.chunk_id for r in old] == ["second", "first"]
    assert [r.chunk_id for r in new] == ["first", "second"]
    assert abs(old[0].dense_score - new[0].dense_score) < 1e-4


def test_1024_dimension_cosine_parity() -> None:
    rng = np.random.default_rng(20261006)
    matrix = rng.normal(size=(80, 1024))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    raw = {str(i): row.tolist() for i, row in enumerate(matrix)}
    chunks = [_chunk(k, "same text") for k in raw]
    packed = _packed(raw)
    for query in rng.normal(size=(4, 1024)).tolist():
        result = LocalDenseRetriever(chunks=chunks, embeddings=packed).search(query_vector=query, top_k=80)
        for r in result:
            # Matmul arithmetic versus Python on identical quantized inputs.
            assert abs(r.dense_score - cosine_similarity(query, packed[r.chunk_id])) < 2e-7
            # Includes float16 storage quantization versus original vectors.
            assert abs(r.dense_score - cosine_similarity(query, raw[r.chunk_id])) < 1e-4


@pytest.mark.parametrize("cutoff,expected", [
    ("2026-01-01T00:00:00+00:00", []),
    ("2026-02-01T00:00:00+00:00", ["original", "unamended"]),
    ("2026-03-01T00:00:00+00:00", ["unamended", "amended"]),
    (None, ["original", "unamended", "amended", "future", "undated"]),
])
def test_packed_as_of_supersession_and_filters(cutoff: str | None, expected: list[str]) -> None:
    from datetime import datetime

    chunks = []
    for chunk_id, filed_at, form, item, ticker in [
        ("original", "2026-02-01T00:00:00+00:00", "10-K", "7", "NVDA"),
        ("unamended", "2026-02-01T00:00:00+00:00", "10-K", "1A", "NVDA"),
        ("amended", "2026-03-01T00:00:00+00:00", "10-K/A", "7", "NVDA"),
        ("future", "2027-01-01T00:00:00+00:00", "10-Q", "7", "NVDA"),
        ("undated", "", "10-Q", "7", "NVDA"),
        ("other", "2026-01-01T00:00:00+00:00", "10-K", "7", "AMD"),
    ]:
        chunk = _chunk(chunk_id, "same text", ticker=ticker)
        chunk.metadata.update(filed_at=filed_at, form_type=form, item_number=item,
                              cik=ticker, period_end="2025-12-31", accession_number=chunk_id)
        chunks.append(chunk)
    raw = {c.chunk_id: [1.0, 0.0] for c in chunks}
    kwargs = dict(query_vector=[1.0, 0.0], top_k=99,
                  filters=RetrievalFilters(ticker="NVDA", document_roles=("primary",)),
                  as_of=datetime.fromisoformat(cutoff) if cutoff else None)
    for embeddings in (raw, _packed(raw)):
        result = LocalDenseRetriever(chunks=chunks, embeddings=embeddings).search(**kwargs)
        assert [r.chunk_id for r in result] == expected
        assert all(r.dense_score == 1.0 for r in result)


def test_dense_batches_keep_candidate_alignment_across_boundary() -> None:
    raw = {f"row-{i}": ([1.0, 0.0] if i % 2 else [0.0, 1.0]) for i in range(4100)}
    chunks = [_chunk(k, "same text") for k in raw]
    chunks.insert(4096, _chunk("missing", "same text"))
    old = LocalDenseRetriever(chunks=chunks, embeddings=raw).search(query_vector=[1.0, 0.0], top_k=5000)
    new = LocalDenseRetriever(chunks=chunks, embeddings=_packed(raw)).search(query_vector=[1.0, 0.0], top_k=5000)
    assert [(r.chunk_id, r.dense_score) for r in new] == [(r.chunk_id, r.dense_score) for r in old]
