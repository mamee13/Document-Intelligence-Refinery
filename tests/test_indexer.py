import pytest

from src.agents.indexer import PageIndexManager
from src.models.core import LDU


@pytest.fixture  # type: ignore[misc]
def mock_ldus() -> list[LDU]:
    return [
        LDU(
            chunk_id="head_1",
            doc_id="test_doc",
            content="Introduction",
            chunk_type="section_header",
            page_refs=[1],
            content_hash="hash1",
        ),
        LDU(
            chunk_id="para_1",
            doc_id="test_doc",
            content="This is the intro text.",
            chunk_type="paragraph",
            page_refs=[1],
            parent_section="Introduction",
            content_hash="hash2",
        ),
        LDU(
            chunk_id="head_2",
            doc_id="test_doc",
            content="Methods",
            chunk_type="section_header",
            page_refs=[2],
            content_hash="hash3",
        ),
        LDU(
            chunk_id="para_2",
            doc_id="test_doc",
            content="This is the methods text.",
            chunk_type="paragraph",
            page_refs=[2],
            parent_section="Methods",
            content_hash="hash4",
        ),
    ]


@pytest.mark.asyncio  # type: ignore[misc]
async def test_page_index_creation(mock_ldus: list[LDU], monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PageIndexManager()

    # Mock the LLM call to avoid actual API requests during tests
    async def mock_summarize(title: str, content: str) -> str:
        return f"Summary of {title}"

    monkeypatch.setattr(manager, "_summarize_section", mock_summarize)

    index = await manager.create_index("test_doc", mock_ldus)

    assert index.doc_id == "test_doc"
    assert len(index.root_nodes) == 2
    assert index.root_nodes[0].title == "Introduction"
    assert index.root_nodes[0].summary == "Summary of Introduction"
    assert index.root_nodes[1].title == "Methods"
    assert index.root_nodes[1].page_start == 2
