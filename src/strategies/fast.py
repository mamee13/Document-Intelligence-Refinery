import time
from pathlib import Path
from typing import List, Optional

import pdfplumber

from src.models.core import BBox, ExtractedDocument, ExtractedText
from src.strategies.base import BaseExtractor


class FastTextExtractor(BaseExtractor):
    """
    Strategy A: Fast text extraction using pdfplumber.
    Best for native digital documents with single-column layouts.
    """

    def extract(
        self, pdf_path: Path, doc_id: str, max_pages: Optional[int] = None
    ) -> ExtractedDocument:
        start_time = time.time()
        text_blocks: List[ExtractedText] = []

        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            for i, page in enumerate(pages):
                page_text = page.extract_text()
                if page_text:
                    # For pdfplumber, we get the whole page text as one block Strategy A simplicity
                    # If needed later, we can extract words for better bbox resolution
                    text_blocks.append(
                        ExtractedText(
                            text=page_text,
                            page_number=i + 1,
                            bbox=BBox(x0=0, y0=0, x1=float(page.width), y1=float(page.height)),
                        )
                    )

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=text_blocks,
            strategy_used="A_FastText",
            extraction_time_seconds=time.time() - start_time,
        )
