"""Offline coverage of pinned artifact loading and the hosted access boundary."""
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from scripts.financial_rag_build_demo_slice import build_slice
from src.financial_rag.hosted_demo import bridge_provider_secrets, load_demo_service, retrieval_notice


def _source(root: Path) -> Path:
    chunks = root / 'data/filings/chunks'
    packed = root / 'data/packed'
    chunks.mkdir(parents=True)
    packed.mkdir(parents=True)
    for ticker in ('NVDA', 'AAPL'):
        row = {'chunk_id': ticker, 'chunk_text': 'Risk factors include export controls and demand.',
               'ticker': ticker, 'document_id': ticker, 'content_hash': 'hash-' + ticker,
               'source_url': 'https://www.sec.gov/Archives/' + ticker + '.htm',
               'form_type': '10-K', 'item_number': '1A', 'filing_date': '2026-01-01',
               'accession_number': '0000000001-26-000001'}
        (chunks / (ticker + '.jsonl')).write_text(json.dumps(row) + '\n', encoding='utf-8')
    np.save(packed / 'vectors.npy', np.array([[1., 0.]], dtype=np.float16))
    (packed / 'chunk_ids.json').write_text('["NVDA"]', encoding='utf-8')
    (packed / 'meta.json').write_text(json.dumps({'count': 1, 'dimensions': 2,
                                                  'dtype': 'float16', 'model': 'fixture'}), encoding='utf-8')
    return packed


def test_slice_preserves_missing_vectors_and_pins_bytes(tmp_path):
    source = tmp_path / 'source'
    packed = _source(source)
    output = tmp_path / 'slice'
    manifest = build_slice(source=source, output=output, tickers=['NVDA', 'AAPL'], packed_source=packed)
    assert manifest['chunk_count'] == 2
    assert manifest['missing_vector_count'] == 1
    service, loaded = load_demo_service(output, use_voyage=False)
    assert len(service.chunks) == 2
    assert loaded['snapshot_id'] == manifest['snapshot_id']
    with pytest.raises(ValueError, match='fresh directory'):
        build_slice(source=source, output=output, tickers=['NVDA'], packed_source=packed)
    path = output / 'data/filings/chunks/NVDA.jsonl'
    path.write_text(path.read_text().replace('export controls', 'modified words'), encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_demo_service(output, use_voyage=False)


def test_slice_rejects_missing_ticker_without_completion_marker(tmp_path):
    source = tmp_path / 'source'
    packed = _source(source)
    output = tmp_path / 'slice'
    with pytest.raises(ValueError, match='absent'):
        build_slice(source=source, output=output, tickers=['UNKNOWN'], packed_source=packed)
    assert not (output / 'demo_manifest.json').exists()


def test_provider_bridge_and_visible_fallback(monkeypatch):
    monkeypatch.delenv('VOYAGE_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert 'lexical only' in retrieval_notice(True)
    bridge_provider_secrets({'VOYAGE_API_KEY': 'test-voyage', 'OPENAI_API_KEY': 'test-openai',
                             'UNRELATED_SECRET': 'do not copy'})
    import os
    assert os.environ['OPENAI_API_KEY'] == 'test-openai'
    assert 'UNRELATED_SECRET' not in os.environ
    assert 'Voyage query' in retrieval_notice(True)
    assert 'switched off' in retrieval_notice(False)


def test_hosted_gate_blocks_loading_and_clears_password(monkeypatch):
    from scripts import financial_rag_brief_view as view
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise RuntimeError('Load boundary reached')

    monkeypatch.setattr(view, '_demo_service', forbidden)
    app = AppTest.from_string('from scripts.financial_rag_brief_view import main\nmain(hosted=True)')
    app.run()
    assert not app.exception
    assert 'configures' in app.error[0].value
    assert calls == []
    app.secrets['APP_PASSWORD'] = 'test-password'
    app.run()
    app.text_input[0].set_value('wrong').run()
    assert calls == []
    assert app.error[0].value == 'Incorrect password.'
    assert '_demo_password' not in app.session_state.filtered_state or not app.session_state['_demo_password']
    app.text_input[0].set_value('test-password').run()
    assert calls == [True]
    assert app.exception[0].message == 'Load boundary reached'
    assert '_demo_password' not in app.session_state.filtered_state


def test_demo_universe_is_per_service_and_does_not_change_legacy_default(tmp_path):
    from src.financial_rag.api import LocalApiError, QueryRequest, build_local_api_service
    source = tmp_path / 'source'
    packed = _source(source)
    path = source / 'data/filings/chunks/AAPL.jsonl'
    path.write_text(path.read_text().replace('AAPL', 'TSLA'), encoding='utf-8')
    output = tmp_path / 'slice'
    build_slice(source=source, output=output, tickers=['TSLA', 'NVDA'], packed_source=packed)
    demo, _ = load_demo_service(output, use_voyage=False)
    request = QueryRequest(ticker='TSLA', question='What are the export controls?')
    assert demo.query(request)['results'][0]['metadata']['ticker'] == 'TSLA'
    legacy = build_local_api_service(root=output, use_voyage=False)
    with pytest.raises(LocalApiError, match='outside'):
        legacy.query(request)
    with pytest.raises(LocalApiError, match='outside'):
        demo.query(QueryRequest(ticker='AAPL', question='Risks?'))


def test_slice_builds_from_json_vectors_and_retains_missing_vector_chunks(tmp_path):
    from src.financial_rag.corpus_snapshot import read_corpus_snapshot
    from src.financial_rag.storage import LocalRagStore

    source = tmp_path / 'source'
    _source(source)
    cache = source / 'data/vector_cache'
    cache.mkdir()
    for chunk_id in ('NVDA', 'OUTSIDE'):
        (cache / (chunk_id + '.json')).write_text(json.dumps({
            'chunk_id': chunk_id, 'embedding': [0.5, 1.0], 'model': 'fixture',
        }), encoding='utf-8')
    output = tmp_path / 'slice'
    manifest = build_slice(source=source, output=output, tickers=['NVDA', 'AAPL'])
    service, loaded = load_demo_service(output, use_voyage=False)
    assert loaded == manifest
    assert manifest['chunk_count'] == 2
    assert manifest['vector_count'] == 1
    assert manifest['missing_vector_count'] == 1
    assert {chunk.chunk_id for chunk in service.chunks} == {'NVDA', 'AAPL'}
    assert set(service.retriever.embeddings) == {'NVDA'}
    assert service.retriever.embeddings['NVDA'] == [0.5, 1.0]
    snapshot = read_corpus_snapshot(LocalRagStore(root=output), manifest['snapshot_id'])
    assert snapshot is not None
    assert snapshot.chunk_count == 2
    assert snapshot.document_ids() == {'NVDA', 'AAPL'}
