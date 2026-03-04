import pytest

from src.agents.chunker import ChunkingEngine
from src.models.core import ExtractedDocument, ExtractedFigure, ExtractedTable, ExtractedText


@pytest.fixture  # type: ignore[misc]
def mock_doc() -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="test_doc",
        strategy_used="B_Layout",
        text_blocks=[
            ExtractedText(text="Main Section Title", page_number=1),
            ExtractedText(text="This is a paragraph under the main section.", page_number=1),
            ExtractedText(text="1. First list item", page_number=2),
            ExtractedText(text="2. Second list item", page_number=2),
            ExtractedText(text="Concluding thoughts.", page_number=2),
        ],
        tables=[
            ExtractedTable(
                markdown_grid="| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |",
                page_number=1,
                caption="Test Table",
            )
        ],
        figures=[ExtractedFigure(caption="Test Figure", page_number=1)],
    )


def test_chunking_rules(mock_doc: ExtractedDocument) -> None:
    engine = ChunkingEngine()
    ldus = engine.chunk(mock_doc)

    # Check total counts
    # 1 table, 1 figure, 1 header, 1 paragraph, 1 list (aggregated), 1 concluding paragraph = 6 LDUs
    assert len(ldus) >= 5

    # Verify Table Integrity (Rule 1)
    table_ldus = [chunk for chunk in ldus if chunk.chunk_type == "table"]
    assert len(table_ldus) == 1
    assert "Table Caption: Test Table" in table_ldus[0].content

    # Verify Figure Captions (Rule 2)
    fig_ldus = [chunk for chunk in ldus if chunk.chunk_type == "figure"]
    assert len(fig_ldus) == 1
    assert "Figure Caption: Test Figure" in fig_ldus[0].content

    # Verify Section Inheritance (Rule 4)
    header_ldus = [chunk for chunk in ldus if chunk.chunk_type == "section_header"]
    assert len(header_ldus) == 1
    assert header_ldus[0].content == "Main Section Title"

    # Next paragraph should have the header as parent
    para_ldus = [chunk for chunk in ldus if chunk.chunk_type == "paragraph"]
    assert any(chunk.parent_section == "Main Section Title" for chunk in para_ldus)

    # Verify List Integrity (Rule 3)
    list_ldus = [chunk for chunk in ldus if chunk.chunk_type == "list"]
    assert len(list_ldus) == 1
    assert "1. First list item" in list_ldus[0].content
    assert "2. Second list item" in list_ldus[0].content
