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

            # 1. Extract words with their bboxes
            for i, page in enumerate(pages):
                words = page.extract_words()
                width = float(page.width or 1)
                height = float(page.height or 1)
                total_area += width * height

                # Track stats for confidence
                page_text = page.extract_text() or ""
                total_chars += len(page_text.strip())

                img_area = 0.0
                for img in page.images:
                    img_area += img.get("width", 0) * img.get("height", 0)
                total_image_area += img_area

                for char in page.chars:
                    font_types.add(char.get("fontname", "unknown"))

                # 2. Simple Line Clustering for BBoxes
                # Group words into blocks (lines) to avoid 1 block per word
                if not words:
                    text_blocks.append(ExtractedText(text=page_text, page_number=i + 1, bbox=None))
                    continue

                # Sort words by top position
                words.sort(key=lambda w: (w["top"], w["x0"]))

                current_block_words = []
                last_bottom = None

                for word in words:
                    # Threshold for starting a new block: vertical gap > 5 pts
                    if last_bottom is not None and (word["top"] - last_bottom) > 5:
                        # Flush current block
                        block_text = " ".join([w["text"] for w in current_block_words])
                        bx0 = min(w["x0"] for w in current_block_words)
                        by0 = min(w["top"] for w in current_block_words)
                        bx1 = max(w["x1"] for w in current_block_words)
                        by1 = max(w["bottom"] for w in current_block_words)

                        bbox = BBox(x0=float(bx0), y0=float(by0), x1=float(bx1), y1=float(by1))
                        text_blocks.append(
                            ExtractedText(text=block_text, page_number=i + 1, bbox=bbox)
                        )
                        current_block_words = []

                    current_block_words.append(word)
                    last_bottom = word["bottom"]

                # Flush remaining
                if current_block_words:
                    block_text = " ".join([w["text"] for w in current_block_words])
                    bx0 = min(w["x0"] for w in current_block_words)
                    by0 = min(w["top"] for w in current_block_words)
                    bx1 = max(w["x1"] for w in current_block_words)
                    by1 = max(w["bottom"] for w in current_block_words)
                    bbox = BBox(
                        x0=float(bx0),
                        y0=float(by0),
                        x1=float(bx1),
                        y1=float(by1),
                    )
                    text_blocks.append(
                        ExtractedText(
                            text=block_text,
                            page_number=i + 1,
                            bbox=bbox,
                        )
                    )

        # Confidence Calculation (Mastered Requirement)
        char_density = total_chars / total_area if total_area > 0 else 0
        image_ratio = total_image_area / total_area if total_area > 0 else 0

        # Heuristic:
        # 1. Native digital usually has >2 fonts and <30% image area
        # 2. Scanned (Strategy A failure) usually has 0 fonts (or OCR fonts)
        #    and >50% image area
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
