import logging
import time
from pathlib import Path
from typing import List, Optional

from docling.datamodel.document import TableItem, TextItem
from docling.document_converter import DocumentConverter

from src.models.core import (
    BBox,
    ExtractedDocument,
    ExtractedTable,
    ExtractedText,
)
from src.strategies.base import BaseExtractor
from src.utils.config import RULES

logger = logging.getLogger(__name__)


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

        try:
            if max_pages:
                result = self.converter.convert(pdf_path, page_range=(1, max_pages))
            else:
                result = self.converter.convert(pdf_path)
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            raise e

        doc = result.document
        text_blocks: List[ExtractedText] = []
        tables: List[ExtractedTable] = []

        # Docling 2.x exports to markdown or structured nodes
        for item, level in doc.iterate_items():
            page_no = 1
            bbox = None
            if hasattr(item, "prov") and item.prov:
                # Docling prov is a list of locations
                loc = item.prov[0]
                page_no = loc.page_no
                if hasattr(loc, "bbox") and loc.bbox:
                    # Docling 2.x bbox has l, t, r, b (normalized/points)
                    bbox = BBox(
                        x0=float(loc.bbox.l),
                        y0=float(loc.bbox.t),
                        x1=float(loc.bbox.r),
                        y1=float(loc.bbox.b),
                    )

            if max_pages and page_no > max_pages:
                continue

            if isinstance(item, TableItem):
                md_table = item.export_to_markdown(doc=doc)
                # Master Thinker: monitor table success
                is_empty = len(item.data.table_cells) == 0
                tables.append(
                    ExtractedTable(
                        markdown_grid=md_table,
                        page_number=page_no,
                        bbox=bbox,
                        caption=f"EmptyTable:{is_empty}" if is_empty else None,
                    )
                )
            elif isinstance(item, TextItem):
                text_blocks.append(ExtractedText(text=item.text, page_number=page_no, bbox=bbox))

        # Sparse content check (Mastered requirement)
        t = RULES["thresholds"]["triage"]
        min_blocks = t.get("min_text_blocks_for_high_conf", 5)

        conf = 0.95
        if not text_blocks and not tables:
            conf = 0.1
        elif len(text_blocks) < min_blocks and not tables:
            conf = 0.6
        elif len(tables) > 0 and not text_blocks:
            conf = 0.85

        # Penalize if many bboxes are missing (facade check)
        missing_bbox = sum(1 for b in text_blocks if b.bbox is None)
        if text_blocks and (missing_bbox / len(text_blocks)) > 0.5:
            conf -= 0.2

        confidence = max(0.1, min(conf, 1.0))

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=text_blocks,
            tables=tables,
            strategy_used="B_Layout",
            confidence_score=confidence,
            extraction_time_seconds=time.time() - start_time,
        )
