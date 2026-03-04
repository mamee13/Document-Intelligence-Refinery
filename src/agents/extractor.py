import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.core import DocumentProfile, ExtractedDocument
from src.strategies.fast import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor
from src.utils.config import RULES

logger = logging.getLogger(__name__)


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
        self.escalation_rules = RULES["thresholds"]["triage"]

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
        return self.layout

    def route_and_extract(
        self, pdf_path: Path, profile: DocumentProfile, max_pages: Optional[int] = None
    ) -> ExtractedDocument:
        """Routes the document with escalation."""
        # 1. Primary Strategy Selection
        strategy = self._select_strategy(profile)

        # 2. Execution with Tier Escalation (A -> B -> C)
        extraction = None

        # Tier A: Fast (Native Text)
        if strategy == self.fast:
            extraction = self.fast.extract(pdf_path, profile.doc_id, max_pages=max_pages)
            if extraction.confidence_score < self.escalation_rules["min_conf_a"]:
                logger.warning(f"Escalating {profile.doc_id} from A to B")
                strategy = self.layout  # Escalate to B

        # Tier B: Layout (Docling / Structured)
        if strategy == self.layout:
            extraction = self.layout.extract(pdf_path, profile.doc_id, max_pages=max_pages)
            if extraction.confidence_score < self.escalation_rules["min_conf_b"]:
                logger.warning(f"Escalating {profile.doc_id} from B to C")
                strategy = self.vision  # Escalate to C

        # Tier C: Vision (VLM / API)
        if strategy == self.vision:
            extraction = self.vision.extract(pdf_path, profile.doc_id, max_pages=max_pages)

        # 3. Graceful Degradation / Human Review Check
        if extraction and extraction.confidence_score < self.escalation_rules["min_conf_c"]:
            extraction.needs_human_review = True
            logger.error(f"Extraction for {profile.doc_id} failed all tiers.")

        # Finalize and log
        if extraction:
            self._log_to_ledger(extraction)
            return extraction

        raise RuntimeError("Extraction failed to produce a result.")
