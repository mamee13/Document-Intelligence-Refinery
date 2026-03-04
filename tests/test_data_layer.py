import os
import shutil
from typing import Any

import pytest

from src.data.fact_table import FactTable
from src.data.vector_store import VectorStore
from src.models.core import LDU


@pytest.fixture  # type: ignore[misc]
def mock_ldus() -> list[LDU]:
    return [
        LDU(
            chunk_id="test_chunk_1",
            doc_id="test_doc",
            content="The annual revenue of CBE in 2023 was 100 billion ETB.",
            chunk_type="paragraph",
            page_refs=[5],
            content_hash="hash123",
        )
    ]


@pytest.fixture  # type: ignore[misc]
def vector_store() -> Any:
    # Setup
    vs = VectorStore(index_name="test_index")
    vs.persist_dir = ".refinery/test_vectorstore"
    yield vs
    # Teardown
    if os.path.exists(vs.persist_dir):
        shutil.rmtree(vs.persist_dir)


@pytest.fixture  # type: ignore[misc]
def fact_table() -> Any:
    db_path = ".refinery/test_facts.db"
    ft = FactTable(db_path=db_path)
    yield ft
    # Teardown
    if os.path.exists(db_path):
        os.remove(db_path)


def test_vector_store_ingest_search(
    vector_store: VectorStore, mock_ldus: list[LDU], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mock embeddings to avoid API calls
    class MockEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1536 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 1536

        def __call__(self, text: str) -> list[float]:
            # Some versions of LangChain might call the object directly
            return self.embed_query(text)

    monkeypatch.setattr(vector_store, "embeddings", MockEmbeddings())

    vector_store.ingest_ldus(mock_ldus)
    results = vector_store.search("revenue", k=1)

    assert len(results) == 1
    assert results[0].page_content == mock_ldus[0].content
    assert results[0].metadata["doc_id"] == "test_doc"


def test_fact_table_insert_query(fact_table: FactTable) -> None:
    fact = {
        "doc_id": "test_doc",
        "fact_key": "Annual Revenue",
        "fact_value": "100 billion",
        "unit": "ETB",
        "confidence": 0.95,
    }
    row_id = fact_table.insert_fact(fact)
    assert row_id > 0

    results = fact_table.query_facts(doc_id="test_doc", fact_key="Revenue")
    assert len(results) == 1
    assert results[0]["fact_value"] == "100 billion"
    assert results[0]["unit"] == "ETB"
