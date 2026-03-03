import time
from pathlib import Path
from typing import List, Optional

from docling.datamodel.document import TableItem, TextItem
from docling.document_converter import DocumentConverter

from src.models.core import ExtractedDocument, ExtractedTable, ExtractedText
from src.strategies.base import BaseExtractor


class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Layout-aware extraction using Docling.
    Best for complex layouts, multi-column text, and tables.
    """

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def extract(
        self, pdf_path: Path, doc_id: str, max_pages: Optional[int] = None
    ) -> ExtractedDocument:
        start_time = time.time()

        # Docling 2.x convert call
        result = self.converter.convert(pdf_path)
        doc = result.document

        text_blocks: List[ExtractedText] = []
        tables: List[ExtractedTable] = []

        # Docling 2.x exports to markdown or structured nodes
        for item, level in doc.iterate_items():
            page_no = 1
            if hasattr(item, "prov") and item.prov:
                # Docling prov is a list of locations
                page_no = item.prov[0].page_no

            # If max_pages is set, we bypass items from later pages
            if max_pages and page_no > max_pages:
                continue

            if isinstance(item, TableItem):
                # Fix deprecation: pass doc argument
                md = item.export_to_markdown(doc=doc)
                tables.append(ExtractedTable(markdown_grid=md, page_number=page_no, bbox=None))
            elif isinstance(item, TextItem):
                text_blocks.append(ExtractedText(text=item.text, page_number=page_no, bbox=None))

        # Basic confidence: if we have text, it's likely good
        conf = 0.95 if text_blocks else 0.5

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=text_blocks,
            tables=tables,
            strategy_used="B_Layout",
            confidence_score=conf,
            extraction_time_seconds=time.time() - start_time,
        )
