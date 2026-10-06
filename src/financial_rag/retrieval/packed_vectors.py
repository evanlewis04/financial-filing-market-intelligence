"""Memory-mapped derived vector cache, with the existing mapping interface."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

VECTORS_FILENAME = "vectors.npy"
CHUNK_IDS_FILENAME = "chunk_ids.json"
META_FILENAME = "meta.json"
DEFAULT_PACKED_DTYPE = "float16"


class PackedVectors(Mapping[str, list[float]]):
    """Read-only mapping over a matrix; bulk scoring uses matrix/row_for."""

    def __init__(
        self,
        *,
        chunk_ids: Sequence[str],
        matrix: np.ndarray,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if matrix.ndim != 2 or matrix.shape[1] < 1:
            raise ValueError(f"Packed vectors must be a 2-D matrix with dimensions, got {matrix.shape}.")
        if matrix.dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("Packed vectors require float16 or float32 storage.")
        if len(chunk_ids) != matrix.shape[0]:
            raise ValueError("chunk_ids length does not match matrix rows.")
        if any(not isinstance(value, str) or not value for value in chunk_ids):
            raise ValueError("chunk_ids must be nonempty strings.")
        self.chunk_ids = list(chunk_ids)
        self._rows = {chunk_id: row for row, chunk_id in enumerate(chunk_ids)}
        if len(self._rows) != len(chunk_ids):
            raise ValueError("Packed chunk_ids must be unique.")
        self.matrix = matrix
        self.meta = dict(meta or {})

    @property
    def dimensions(self) -> int:
        return int(self.matrix.shape[1])

    def row_for(self, chunk_id: str) -> int | None:
        return self._rows.get(chunk_id)

    def __getitem__(self, chunk_id: str) -> list[float]:
        return self.matrix[self._rows[chunk_id]].astype(float).tolist()

    def __contains__(self, chunk_id: object) -> bool:
        # Readiness probes every chunk; membership must not materialize rows.
        return chunk_id in self._rows

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def pack_vector_cache(
    *,
    vector_cache_dir: Path | str,
    out_dir: Path | str,
    chunk_ids: Iterable[str] | None = None,
    dtype: str = DEFAULT_PACKED_DTYPE,
) -> dict[str, Any]:
    """Pack a cache into a fresh output directory.

    The optional id allowlist supports demo slices without importing chunk
    metadata here. Float16 quantization can change close dense rankings; this
    is a storage tradeoff, not a guarantee of exact dense ranking equivalence.
    Malformed/incompatible vectors fail the pack rather than losing evidence.
    Rebuild to a new directory; an existing artifact is never overwritten.
    """
    if dtype not in ("float16", "float32"):
        raise ValueError("dtype must be float16 or float32.")
    source_dir, target_dir = Path(vector_cache_dir), Path(out_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Vector cache directory not found: {source_dir}")
    filenames = (VECTORS_FILENAME, CHUNK_IDS_FILENAME, META_FILENAME)
    if any((target_dir / name).exists() for name in filenames):
        raise FileExistsError(f"Use a fresh output directory: {target_dir}")
    allowlist = set(chunk_ids) if chunk_ids is not None else None
    rows: dict[str, int] = {}
    vectors: list[np.ndarray] = []
    dimensions = 0
    models: set[str] = set()
    duplicate_count = 0
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a vector object in {path}")
        chunk_id = str(payload.get("chunk_id", ""))
        vector = payload.get("embedding", [])
        # Match the fallback loader's handling of non-vector records.
        if not chunk_id or not isinstance(vector, list):
            continue
        if allowlist is not None and chunk_id not in allowlist:
            continue
        if not vector:
            raise ValueError(f"Cannot pack an empty vector: {path}")
        if dimensions and len(vector) != dimensions:
            raise ValueError(f"Mixed vector dimensions in {path}; retain the JSON fallback.")
        dimensions = len(vector)
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray([float(value) for value in vector], dtype=dtype)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite or out-of-range {dtype} vector in {path}")
        model = str(payload.get("model", ""))
        if model:
            models.add(model)
        if len(models) > 1:
            raise ValueError("Mixed embedding models; retain the JSON fallback.")
        if chunk_id in rows:
            # Sorted JSON traversal is last-file-wins in _load_embeddings.
            vectors[rows[chunk_id]] = array
            duplicate_count += 1
        else:
            rows[chunk_id] = len(vectors)
            vectors.append(array)
    matrix = np.stack(vectors) if vectors else np.zeros((0, 1), dtype=dtype)
    meta = {
        "count": len(rows),
        "dimensions": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "model": next(iter(models), "unknown"),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_dir),
        "restricted_to_chunk_ids": allowlist is not None,
        "duplicate_chunk_ids": duplicate_count,
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    # Readers require all three files. Publish metadata last, only after a
    # complete pack; failures while generating leave no visible artifact.
    with tempfile.TemporaryDirectory(prefix=".pack-", dir=target_dir) as temp:
        staging = Path(temp)
        np.save(staging / VECTORS_FILENAME, matrix, allow_pickle=False)
        (staging / CHUNK_IDS_FILENAME).write_text(json.dumps(list(rows)), encoding="utf-8")
        (staging / META_FILENAME).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        for name in filenames:
            os.replace(staging / name, target_dir / name)
    return meta


def packed_vectors_exist(packed_dir: Path | str) -> bool:
    directory = Path(packed_dir)
    return all((directory / name).is_file() for name in (VECTORS_FILENAME, CHUNK_IDS_FILENAME, META_FILENAME))


def load_packed_vectors(packed_dir: Path | str) -> PackedVectors:
    directory = Path(packed_dir)
    if not packed_vectors_exist(directory):
        raise FileNotFoundError(f"No complete packed vectors under {directory}")
    chunk_ids = json.loads((directory / CHUNK_IDS_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(chunk_ids, list):
        raise ValueError("Expected a JSON list of chunk ids.")
    meta = json.loads((directory / META_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Expected a packed metadata object.")
    matrix = np.load(directory / VECTORS_FILENAME, mmap_mode="r", allow_pickle=False)
    packed = PackedVectors(chunk_ids=chunk_ids, matrix=matrix, meta=meta)
    if (meta.get("count"), meta.get("dimensions"), meta.get("dtype")) != (
        len(packed), packed.dimensions, str(matrix.dtype)
    ):
        raise ValueError("Packed metadata does not match the ids and matrix.")
    return packed
