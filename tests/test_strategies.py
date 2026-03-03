from pathlib import Path

import pytest

from src.models.core import ExtractedDocument
from src.strategies.fast import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor

# Sample document for testing (use one from data/ if available)
SAMPLE_PDF = Path("data/tax_expenditure_ethiopia_2021_22.pdf")


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not found")  # type: ignore[misc]
def test_fast_extractor() -> None:
    extractor = FastTextExtractor()
    doc = extractor.extract(SAMPLE_PDF, "test_fast", max_pages=1)
    assert isinstance(doc, ExtractedDocument)
    assert doc.strategy_used == "A_FastText"
    assert len(doc.text_blocks) > 0


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not found")  # type: ignore[misc]
def test_layout_extractor() -> None:
    # Docling might take time and requires its own dependencies
    extractor = LayoutExtractor()
    doc = extractor.extract(SAMPLE_PDF, "test_layout", max_pages=1)
    assert isinstance(doc, ExtractedDocument)
    assert doc.strategy_used == "B_Layout"
    # Layout extractor should find both text and likely tables
    assert len(doc.text_blocks) > 0


def test_vision_extractor_no_key() -> None:
    # Test fallback behavior when no API key is present
    extractor = VisionExtractor()
    extractor.api_key = None
    doc = extractor.extract(SAMPLE_PDF, "test_vision", max_pages=1)
    assert "[MOCK VISION OUTPUT: No API Key]" in doc.text_blocks[0].text
