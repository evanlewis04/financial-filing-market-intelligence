"""Exact selected-chunk evidence delivery, independent of compact UI excerpts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class EvidenceDeliveryError(ValueError):
    """The selected evidence cannot safely be sent to the answer provider."""


@dataclass(frozen=True)
class FullTextEvidence:
    chunk_id: str
    label: str
    text: str
    source_url: str
    metadata: dict[str, Any]


def validate_evidence_identity(result: dict[str, Any], evidence: FullTextEvidence) -> None:
    metadata = {k: v for k, v in result.get('metadata', {}).items() if k != 'citation_label'}
    if (str(result.get('chunk_id', '')) != evidence.chunk_id
            or str(result.get('citation_label', '')) != evidence.label
            or str(result.get('source_url', '')) != evidence.source_url
            or metadata != evidence.metadata
            or not evidence.text.strip()):
        raise EvidenceDeliveryError('Selected source identity or metadata does not match the pinned chunk.')
    if result.get('metadata', {}).get('citation_label', evidence.label) != evidence.label:
        raise EvidenceDeliveryError('Selected source citation label is inconsistent.')


def resolve_full_text_evidence(query_payload: dict[str, Any], chunks: Iterable[Any]) -> list[FullTextEvidence]:
    results = query_payload.get('results', [])
    ids = [str(row.get('chunk_id', '')) for row in results]
    labels = [str(row.get('citation_label', '')) for row in results]
    if (not ids or any(not x for x in ids + labels)
            or len(set(ids)) != len(ids) or len(set(labels)) != len(labels)):
        raise EvidenceDeliveryError('Selected sources have missing or ambiguous identities/citation labels.')
    wanted = set(ids)
    found = {}
    for chunk in chunks:
        if chunk.chunk_id in wanted:
            if chunk.chunk_id in found:
                raise EvidenceDeliveryError('Selected chunk ID is ambiguous in the pinned corpus.')
            found[chunk.chunk_id] = chunk
    if set(found) != wanted:
        raise EvidenceDeliveryError('A selected source is missing from the loaded pinned corpus.')
    evidence = []
    for result, chunk_id, label in zip(results, ids, labels, strict=True):
        chunk = found[chunk_id]
        block = FullTextEvidence(chunk_id, label, chunk.chunk_text,
                                 str(chunk.metadata.get('source_url', '')), dict(chunk.metadata))
        validate_evidence_identity(result, block)
        evidence.append(block)
    return evidence
