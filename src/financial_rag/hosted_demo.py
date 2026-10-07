"""Pinned demo loading and Streamlit access configuration."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.financial_rag.api import build_local_api_service
from src.financial_rag.corpus_snapshot import read_corpus_snapshot
from src.financial_rag.settings import configured_secret
from src.financial_rag.storage import LocalRagStore


def bridge_provider_secrets(secrets: Mapping[str, Any]) -> None:
    """Copy only server-owned provider keys, never browser input, to the SDK environment."""
    for key in ('OPENAI_API_KEY', 'VOYAGE_API_KEY'):
        value = secrets.get(key)
        if isinstance(value, str) and value.strip():
            os.environ[key] = value.strip()


def require_password(st: Any, secrets: Mapping[str, Any]) -> bool:
    """Fail closed before loading data or invoking providers; retain no entered password."""
    expected = secrets.get('APP_PASSWORD')
    if not isinstance(expected, str) or not expected.strip():
        st.error('Demo access is unavailable until the owner configures its password.')
        return False
    fingerprint = hashlib.sha256(expected.encode()).hexdigest()
    if st.session_state.get('_authenticated_password') == fingerprint:
        if st.sidebar.button('Sign out'):
            st.session_state.clear()
            st.rerun()
        return True

    def check_password() -> None:
        entered = st.session_state.pop('_demo_password', '')
        if hmac.compare_digest(str(entered).encode(), expected.encode()):
            st.session_state['_authenticated_password'] = fingerprint
            st.session_state.pop('_password_incorrect', None)
        else:
            st.session_state['_password_incorrect'] = True

    st.title('Filing Intelligence')
    st.text_input('Demo password', type='password', key='_demo_password', on_change=check_password)
    if st.session_state.get('_password_incorrect'):
        st.error('Incorrect password.')
    return False


def retrieval_notice(use_voyage: bool) -> str:
    if use_voyage and configured_secret('VOYAGE_API_KEY'):
        return 'Retrieval: Voyage query embeddings plus lexical matching.'
    reason = 'Voyage key is not configured' if use_voyage else 'Voyage is switched off'
    return f'Retrieval: lexical only ({reason}); dense scores are disabled.'


def load_demo_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / 'demo_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('tickers') or not manifest.get('files'):
        raise ValueError('Incomplete demo manifest.')
    actual = {p.relative_to(root).as_posix() for p in (root / 'data').rglob('*') if p.is_file()}
    if actual != set(manifest['files']):
        raise ValueError('Demo file inventory differs from the pinned artifact.')
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Artifact path escapes demo root.')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise ValueError(f'Demo artifact hash mismatch: {name}')
    return manifest


def load_demo_service(root: Path, *, use_voyage: bool = True):
    """Actual application startup path, also used by fresh-process measurements."""
    manifest = load_demo_manifest(root)
    snapshot = read_corpus_snapshot(LocalRagStore(root=root), manifest['snapshot_id'])
    if snapshot is None:
        raise ValueError('Pinned corpus snapshot is missing.')
    service = build_local_api_service(root=root, use_voyage=use_voyage, snapshot=snapshot,
                                      supported_tickers=set(manifest["tickers"]))
    if len(service.chunks) != manifest['chunk_count'] or len(service.retriever.embeddings) != manifest['vector_count']:
        raise ValueError('Pinned corpus counts differ from manifest.')
    return service, manifest
