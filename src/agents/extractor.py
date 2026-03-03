import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.core import DocumentProfile, ExtractedDocument
from src.strategies.fast import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor
from src.utils.config import RULES


class ExtractionRouter:
    """
    Orchestrates the extraction process by selecting the appropriate strategy
    based on the DocumentProfile and handling the Escalation Guard.
    """

    def __init__(self) -> None:
        self.fast = FastTextExtractor()
        self.layout = LayoutExtractor()
        self.vision = VisionExtractor()
        self.ledger_path = Path(".refinery/extraction_ledger.jsonl")
        self.escalation_rules = RULES["thresholds"]["escalation"]

    def _log_to_ledger(self, extraction: ExtractedDocument) -> None:
        """Logs an extraction event to the JSONL ledger."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "doc_id": extraction.doc_id,
            "strategy_used": extraction.strategy_used,
            "confidence_score": extraction.confidence_score,
            "cost_estimate": extraction.cost_estimate,
            "processing_time": extraction.extraction_time_seconds,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _select_strategy(self, profile: DocumentProfile) -> object:
        """Selects the appropriate strategy based on the DocumentProfile."""
        if profile.origin_type == "scanned_image":
            return self.vision
        elif profile.layout_complexity == "single_column":
            return self.fast
        else:
            return self.layout

    def route_and_extract(
        self, pdf_path: Path, profile: DocumentProfile, max_pages: Optional[int] = None
    ) -> ExtractedDocument:
        """
        Routes the document to a strategy and performs extraction with escalation guard.
        """
        # 1. Primary Strategy Selection
        strategy = self._select_strategy(profile)

        if strategy == self.fast:
            extraction = self.fast.extract(pdf_path, profile.doc_id, max_pages=max_pages)

            # Simple confidence for FastText: high if text exists
            extraction.confidence_score = 0.98 if extraction.text_blocks else 0.1

            # 2. Escalation Guard: Check if FastText confidence is low
            num_pages = len(extraction.text_blocks)
            avg_len = (
                sum(len(b.text) for b in extraction.text_blocks) / num_pages if num_pages else 0
            )
            if avg_len < self.escalation_rules["min_chars_per_page"]:
                # Escalate to Strategy B
                extraction = self.layout.extract(pdf_path, profile.doc_id, max_pages=max_pages)
                extraction.confidence_score = 0.85
        elif strategy == self.vision:
            extraction = self.vision.extract(pdf_path, profile.doc_id, max_pages=max_pages)
        else:
            extraction = self.layout.extract(pdf_path, profile.doc_id, max_pages=max_pages)

        # Finalize and log
        self._log_to_ledger(extraction)
        return extraction
