"""Idempotent local filesystem storage for Phase 1 filings artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageWriteResult:
    """Result of an idempotent write."""

    path: Path
    created: bool


class LocalRagStore:
    """Write raw, parsed, chunk, and embedding artifacts under ignored data paths."""

    def __init__(self, *, root: Path | str = Path(".")) -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "data" / "filings" / "raw"
        self.parsed_dir = self.root / "data" / "filings" / "parsed"
        self.chunks_dir = self.root / "data" / "filings" / "chunks"
        self.vector_cache_dir = self.root / "data" / "vector_cache"
        self.packed_dir = self.root / "data" / "packed"
        self.snapshots_dir = self.root / "data" / "filings" / "snapshots"
        for directory in (
            self.raw_dir,
            self.parsed_dir,
            self.chunks_dir,
            self.vector_cache_dir,
            self.packed_dir,
            self.snapshots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def raw_path(self, ticker: str, accession_number: str, document_name: str) -> Path:
        return self.raw_dir / ticker.upper() / accession_number.replace("-", "") / document_name

    def parsed_path(self, document_id: str) -> Path:
        return self.parsed_dir / f"{document_id}.txt"

    def chunks_path(self, document_id: str) -> Path:
        return self.chunks_dir / f"{document_id}.jsonl"

    def embedding_path(self, chunk_id: str) -> Path:
        return self.vector_cache_dir / f"{chunk_id}.json"

    def snapshot_manifest_path(self) -> Path:
        """JSONL manifest holding one row per recorded corpus snapshot."""

        return self.snapshots_dir / "manifest.jsonl"

    def write_bytes_once(self, path: Path, content: bytes) -> StorageWriteResult:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return StorageWriteResult(path=path, created=False)
        path.write_bytes(content)
        return StorageWriteResult(path=path, created=True)

    def write_text_once(self, path: Path, content: str) -> StorageWriteResult:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return StorageWriteResult(path=path, created=False)
        path.write_text(content, encoding="utf-8")
        return StorageWriteResult(path=path, created=True)

    def write_json_once(self, path: Path, payload: dict[str, Any]) -> StorageWriteResult:
        return self.write_text_once(path, json.dumps(payload, indent=2, sort_keys=True))

    def write_jsonl_once(self, path: Path, rows: list[dict[str, Any]]) -> StorageWriteResult:
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        return self.write_text_once(path, content)

    def upsert_manifest(
        self,
        manifest_path: Path,
        *,
        key: str,
        record: dict[str, Any],
    ) -> bool:
        """Insert or replace one JSONL manifest row keyed by `key`.

        Returns True when the manifest file content changed.
        """

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rows = read_jsonl(manifest_path)
        before = json.dumps(rows, sort_keys=True)
        by_key = {str(row.get(key, "")): row for row in rows if row.get(key)}
        by_key[str(record[key])] = record
        updated_rows = [by_key[row_key] for row_key in sorted(by_key)]
        after = json.dumps(updated_rows, sort_keys=True)
        if before == after:
            return False
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in updated_rows),
            encoding="utf-8",
        )
        return True

    def append_manifest(self, manifest_path: Path, record: dict[str, Any]) -> None:
        """Append one JSONL manifest row in O(1), without reading the existing file.

        `upsert_manifest` reads, re-serializes, and rewrites the whole manifest per
        call — fine for the small raw/parsed/chunk manifests, but O(n^2) over a run
        on the embedding manifest, which holds one row (with its full vector) per
        embedded chunk and grows into the hundreds of MB. On that append-mostly hot
        path the caller guards each append with the idempotent per-chunk file write,
        so no duplicate key is introduced, and the manifest stays fully rebuildable
        from the per-chunk files. Row order is not significant to any reader.
        """
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
