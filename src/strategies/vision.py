import base64
import os
import time
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv

from src.models.core import ExtractedDocument, ExtractedText
from src.strategies.base import BaseExtractor
from src.utils.config import RULES

load_dotenv()


class VisionExtractor(BaseExtractor):
    """
    Strategy C: Vision-based extraction using Gemini Flash (via OpenRouter).
    Best for scanned documents, handwriting, or extremely complex layouts.
    Includes a budget guard to prevent excessive API spend.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.vlm_config = RULES["vlm"]
        self.max_cost = self.vlm_config["max_cost_per_doc_usd"]
        self.model = self.vlm_config["model_name"]

    def _pdf_to_images(self, pdf_path: Path) -> List[str]:
        """Converts PDF pages to base64 encoded PNG images."""
        images = []
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            images.append(base64.b64encode(img_data).decode("utf-8"))
        doc.close()
        return images

    def extract(
        self, pdf_path: Path, doc_id: str, max_pages: Optional[int] = None
    ) -> ExtractedDocument:
        start_time = time.time()

        images = self._pdf_to_images(pdf_path)
        if max_pages:
            images = images[:max_pages]

        if not self.api_key:
            # Fallback for demonstration when no API key is present
            total_estimated_cost = len(images) * 0.01
            return ExtractedDocument(
                doc_id=doc_id,
                text_blocks=[ExtractedText(text="[MOCK VISION OUTPUT: No API Key]", page_number=1)],
                strategy_used="C_Vision",
                confidence_score=0.0,
                cost_estimate=total_estimated_cost,
                extraction_time_seconds=time.time() - start_time,
            )

        text_blocks: List[ExtractedText] = []
        total_estimated_cost = 0.0

        for i, img_b64 in enumerate(images):
            # Check budget guard
            if total_estimated_cost >= self.max_cost:
                text_blocks.append(ExtractedText(text="[BUDGET LIMIT REACHED]", page_number=i + 1))
                continue

            # Call VLM for each page
            # Note: In a production system, we'd use batching or parallel requests
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.vlm_config["model_name"],
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Extract all text and tables from this page in Markdown format. Preserving spatial order.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                    },
                                ],
                            }
                        ],
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Update cost estimate (simplified)
                # In real usage, you'd parse data["usage"] and calculate precisely
                total_estimated_cost += 0.01  # Assume $0.01 per image/page

                text_blocks.append(ExtractedText(text=content, page_number=i + 1, bbox=None))
            except Exception as e:
                text_blocks.append(
                    ExtractedText(text=f"[Error processing page: {str(e)}]", page_number=i + 1)
                )

        return ExtractedDocument(
            doc_id=doc_id,
            text_blocks=text_blocks,
            strategy_used="C_Vision",
            confidence_score=0.9,
            cost_estimate=total_estimated_cost,
            extraction_time_seconds=time.time() - start_time,
        )
