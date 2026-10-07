"""Offline actual brief-to-provider input checks using the reviewed pinned cases."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.financial_rag.api import LocalRagApiService
from src.financial_rag.integration import build_unified_brief
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever

CASES = json.loads((Path(__file__).parent / 'fixtures/demo_synthesis_cases.json').read_text(encoding='utf-8'))


class CapturingClient:
    def __init__(self):
        self.calls = []

    def create_response(self, **kwargs):
        self.calls.append(kwargs)
        return 'Offline prompt capture only [S1].'


def case_service(case):
    chunks = [LocalChunkRecord(c['chunk_id'], c['text'], c['metadata']) for c in case['chunks']]
    service = LocalRagApiService(chunks=chunks, retriever=LocalDenseRetriever(chunks=chunks, embeddings={}))
    service.query = lambda request: copy.deepcopy(case['payload'])
    return service


@pytest.mark.parametrize('case', CASES, ids=[c['case_id'] for c in CASES])
def test_actual_brief_prompt_contains_complete_selected_sources(case, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-capture-key')
    before = copy.deepcopy(case['payload'])
    service = case_service(case)
    client = CapturingClient()
    brief = build_unified_brief(service, question=case['question'], ticker=case['ticker'],
                                run_answer=True, openai_client=client).to_dict()
    assert brief['answer_gate']['allowed']
    assert len(client.calls) == 1
    prompt = client.calls[0]['input_text']
    positions = []
    for result, chunk in zip(case['payload']['results'], case['chunks'], strict=True):
        marker = '[' + result['citation_label'] + ']'
        block = prompt.split(marker + '\n', 1)[1].split('\n\n[S', 1)[0]
        assert chunk['text'] in block
        assert 'chunk_id: ' + chunk['chunk_id'] in block
        assert 'filing_date: ' + chunk['metadata']['filing_date'] in block
        assert 'accession: ' + chunk['metadata']['accession_number'] in block
        assert 'url: ' + chunk['metadata']['source_url'] in block
        positions.append(prompt.index(marker + '\n'))
    assert positions == sorted(positions)
    assert case['payload'] == before  # compact retrieval payload is untouched
    if case['case_id'] == 'aapl-gross-margin':
        assert all(word in prompt.lower() for word in ['tariff', 'mix', 'cost'])
        gold = next(c for c in case['chunks'] if '-chunk-0062-' in c['chunk_id'])
        assert len(gold['text']) > 900
        assert gold['text'][900:] in prompt


@pytest.mark.parametrize('failure', ['missing', 'ambiguous', 'metadata', 'duplicate_label', 'oversized'])
def test_evidence_failure_blocks_actual_brief_before_provider(failure, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-capture-key')
    case = copy.deepcopy(CASES[0])
    service = case_service(case)
    if failure == 'missing':
        service.chunks.pop()
    elif failure == 'ambiguous':
        service.chunks.append(service.chunks[0])
    elif failure == 'metadata':
        case['payload']['results'][0]['metadata']['filing_date'] = '1900-01-01'
    elif failure == 'duplicate_label':
        case['payload']['results'][1]['citation_label'] = 'S1'
    elif failure == 'oversized':
        service.chunks[0] = replace(service.chunks[0], chunk_text='x' * 18_001)
    client = CapturingClient()
    brief = build_unified_brief(service, question=case['question'], ticker=case['ticker'],
                                run_answer=True, openai_client=client).to_dict()
    assert not client.calls
    assert brief['answer'] is None
    assert not brief['answer_gate']['allowed']
    assert brief['answer_gate']['reasons']


def test_budget_boundary_accepts_exact_limit_then_blocks_one_more_character(monkeypatch):
    from src.financial_rag.synthesis.openai_responses import (
        MAX_FULL_INPUT_CHARS, _full_evidence_blocks, _input_text, _instructions,
    )
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-capture-key')
    case = copy.deepcopy(CASES[0])
    service = case_service(case)
    evidence = service.synthesis_evidence(case['payload'])
    size = len(_instructions()) + len(_input_text(question=case['question'],
                evidence=_full_evidence_blocks(case['payload'], evidence)))
    original = service.chunks[0]
    service.chunks[0] = replace(original, chunk_text=original.chunk_text + 'x' * (MAX_FULL_INPUT_CHARS - size))
    client = CapturingClient()
    brief = build_unified_brief(service, question=case['question'], ticker=case['ticker'],
                                run_answer=True, openai_client=client)
    assert brief.answer_gate['allowed'] and len(client.calls) == 1
    assert len(client.calls[0]['instructions']) + len(client.calls[0]['input_text']) == MAX_FULL_INPUT_CHARS
    service.chunks[0] = replace(service.chunks[0], chunk_text=service.chunks[0].chunk_text + 'x')
    brief = build_unified_brief(service, question=case['question'], ticker=case['ticker'],
                                run_answer=True, openai_client=client)
    assert not brief.answer_gate['allowed'] and len(client.calls) == 1


def test_hosted_ui_displays_budget_block_without_provider_call(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from scripts import financial_rag_brief_view as view
    from src.financial_rag.synthesis import openai_responses
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-capture-key')
    case = copy.deepcopy(CASES[0])
    service = case_service(case)
    service.chunks[0] = replace(service.chunks[0], chunk_text='x' * 18_001)
    manifest = {'tickers': [case['ticker']], 'source_company_count': 503,
                'built_at': '2026-10-06', 'latest_filing_date': '2026-08-05', 'missing_vector_count': 0}
    monkeypatch.setattr(view, '_demo_service', lambda *args: (service, manifest))
    calls = []

    def forbidden_client():
        calls.append(True)
        raise AssertionError('Provider must not be constructed')

    monkeypatch.setattr(openai_responses, 'OpenAIResponsesClient', forbidden_client)
    app = AppTest.from_string('from scripts.financial_rag_brief_view import main\nmain(hosted=True)')
    app.secrets['APP_PASSWORD'] = 'offline-ui-password'
    app.run()
    app.text_input[0].set_value('offline-ui-password').run()
    for item in app.checkbox:
        if 'Generate grounded answer' in item.label:
            item.set_value(True)
    for item in app.button:
        if item.label == 'Build Brief':
            item.click()
    app.run()
    assert not app.exception and not calls
    assert app.session_state['last_brief']['answer'] is None
    assert any('reduce Top-k or narrow the question' in item.value for item in app.markdown)
