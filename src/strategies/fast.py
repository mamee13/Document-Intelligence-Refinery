import time
from pathlib import Path
from typing import List, Optional

import pdfplumber

from src.models.core import BBox, ExtractedDocument, ExtractedText
from src.strategies.base import BaseExtractor
from src.utils.config import RULES


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

            # Confidence Signals
            total_chars = 0
            total_area = 0.0
            total_image_area = 0.0
            font_types = set()

            for i, page in enumerate(pages):
                page_text = page.extract_text() or ""
                char_count = len(page_text.strip())
                total_chars += char_count

                width = page.width or 1
                height = page.height or 1
                total_area += float(width * height)

                img_area = 0.0
                for img in page.images:
                    img_area += img.get("width", 0) * img.get("height", 0)
                total_image_area += img_area

                for char in page.chars:
                    font_types.add(char.get("fontname", "unknown"))

                text_blocks.append(
                    ExtractedText(
                        text=page_text,
                        page_number=i + 1,
                        bbox=BBox(x0=0, y0=0, x1=float(width), y1=float(height)),
                    )
                )

        # Confidence Calculation (Mastered Requirement)
        char_density = total_chars / total_area if total_area > 0 else 0
        image_ratio = total_image_area / total_area if total_area > 0 else 0

        # Heuristic:
        # 1. Native digital usually has >2 fonts and <30% image area
        # 2. Scanned (Strategy A failure) usually has 0 fonts (or OCR fonts) and >50% image area
        conf = 1.0

        # Penalize for high image area
        if image_ratio > 0.4:
            conf -= 0.4

        # Penalize for lack of fonts (common in non-OCRed scans being read by A)
        if len(font_types) == 0:
            conf -= 0.6
        elif len(font_types) < 2:
            conf -= 0.2
        # Penalize for suspiciously low character density
        t = RULES["thresholds"]["triage"]
        if char_density < t["native_density_min"]:
            conf -= 0.3

        # Penalize for zero-char pages if there are multiple pages
        pages_count = len(pages)
        zero_char_pages = sum(1 for p in pages if len((p.extract_text() or "").strip()) == 0)
        if pages_count > 0 and (zero_char_pages / pages_count) > 0.1:
            conf -= 0.2

        if total_chars == 0:
            conf = 0.0

        confidence = max(0.0, min(conf, 1.0))

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=text_blocks,
            strategy_used="A_FastText",
            confidence_score=confidence,
            extraction_time_seconds=time.time() - start_time,
        )
