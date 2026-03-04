import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.agents.refinery import SemanticRefinery
from src.data.fact_table import FactTableExtractor
from src.models.core import (
    DocumentProfile,
    ExtractedDocument,
    ExtractedTable,
    ExtractedText,
)


@pytest.fixture  # type: ignore[misc]
def refinery() -> Any:
    # Setup directories
    for d in [".refinery/test_vectorstore", ".refinery/pageindex", ".refinery/test_profiles"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    ref = SemanticRefinery()
    # Use test paths
    ref.vector_store.persist_dir = ".refinery/test_vectorstore"
    ref.fact_table.db_path = Path(".refinery/test_facts.db")
    ref.fact_table._init_db()  # Re-init on the test path

    yield ref

    # Teardown
    cleanup_dirs = [".refinery/test_vectorstore", ".refinery/pageindex"]
    for d in cleanup_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    if os.path.exists(".refinery/test_facts.db"):
        os.remove(".refinery/test_facts.db")


@pytest.mark.asyncio  # type: ignore[misc]
async def test_full_refinery_flow(
    refinery: SemanticRefinery, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Mock the API calls (Vision and Indexer)
    async def mock_summarize(self_idx: Any, title: str, content: str) -> str:
        return f"Summary for {title}"

    async def mock_extract_facts(self_idx: Any, doc_id: str, table: Any) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": doc_id,
                "page_number": table.page_number,
                "fact_key": "Extracted Table",
                "fact_value": "Mock Value",
                "unit": "Mock Unit",
                "confidence": 1.0,
                "source_chunk_hash": "hash",
            }
        ]

    def mock_extract_vision(
        self_extractor: Any, pdf_path: Path, doc_id: str, max_pages: int | None = None
    ) -> ExtractedDocument:
        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=[
                ExtractedText(text="Introduction", page_number=1),
                ExtractedText(text="This is a test document.", page_number=1),
                ExtractedText(text="Analysis", page_number=2),
                ExtractedText(text="Data points follow.", page_number=2),
            ],
            tables=[ExtractedTable(markdown_grid="| A | B |\n|---|---|\n| 1 | 2 |", page_number=2)],
            strategy_used="C_Vision",
        )

    # Mock Triage to return a specific profile (forces Vision)
    def mock_classify(self_triage: Any, pdf_path: Path) -> Any:
        return DocumentProfile(
            doc_id=pdf_path.stem, origin_type="scanned_image", layout_complexity="single_column"
        )

    # Mock Embeddings
    class MockEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1536 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 1536

        def __call__(self, text: str) -> list[float]:
            return self.embed_query(text)

    from src.agents.indexer import PageIndexManager
    from src.agents.triage import TriageAgent
    from src.strategies.vision import VisionExtractor

    monkeypatch.setattr(PageIndexManager, "_summarize_section", mock_summarize)
    monkeypatch.setattr(VisionExtractor, "extract", mock_extract_vision)
    monkeypatch.setattr(TriageAgent, "classify_document", mock_classify)
    monkeypatch.setattr(refinery.vector_store, "embeddings", MockEmbeddings())
    monkeypatch.setattr(FactTableExtractor, "extract_facts_from_table", mock_extract_facts)

    # 2. Run the refinery
    test_pdf = Path("data/simple_test.pdf")
    # Create simple_test.pdf if not exists using fitz
    if not test_pdf.exists():
        import fitz

        test_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "Mock Scan Data")
        doc.save(str(test_pdf))
        doc.close()

    index = await refinery.refine_document(test_pdf, max_pages=2)

    # 3. Assertions
    assert index.doc_id == "simple_test"
    assert len(index.root_nodes) == 2
    assert index.root_nodes[0].title == "Introduction"

    # Check Vector Store persist
    assert os.path.exists(refinery.vector_store.persist_dir)
    results = refinery.vector_store.search("test", k=1)
    assert len(results) > 0

    # Check PageIndex navigation
    nav_results = await refinery.indexer.navigate(
        "simple_test", "Introduction", k=1, vector_store=refinery.vector_store
    )
    assert len(nav_results) > 0
    assert nav_results[0].title == "Introduction"

    # Check Fact Table
    facts = refinery.fact_table.query_facts(doc_id="simple_test")
    assert len(facts) == 1
    assert "Extracted Table" in facts[0]["fact_key"]
