"""Hosted entrypoint: password is mandatory, including on configuration failure."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.financial_rag_brief_view import main

main(hosted=True)
