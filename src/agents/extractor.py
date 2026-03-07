import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

from src.models.core import DocumentProfile, ExtractedDocument
from src.strategies.base import BaseExtractor
from src.strategies.fast import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor
from src.utils.config import RULES

logger = logging.getLogger(__name__)


class ExtractionRouter:
    """
    Orchestrates the extraction process by selecting the appropriate strategy
    based on the DocumentProfile and handling the Escalation Guard.
    Upgraded for Per-Page Escalation and Real Cost Tracking.
    """

    def __init__(self) -> None:
        self.fast = FastTextExtractor()
        self.layout = LayoutExtractor()
        self.vision = VisionExtractor()
        self.ledger_path = Path(".refinery/extraction_ledger.jsonl")
        self.rules: Dict[str, Any] = RULES

    def _calculate_actual_cost(self, extraction: ExtractedDocument) -> float:
        """Calculates precise cost based on extraction rules."""
        vlm_rules: Dict[str, Any] = self.rules.get("vlm", {})

        # Strategy A/B are usually local/free unless specific OCR is used
        if "Vision" in extraction.strategy_used:
            # If strategy already calculated it (VisionExtractor does), use it
            if extraction.cost_estimate > 0:
                return float(extraction.cost_estimate)

            # Fallback/Manual calculation
            # Gemini 2.0 Flash is roughly $0.10 / 1M tokens
            # We estimate 1000 tokens per page for structured output
            pages = len(set(b.page_number for b in extraction.text_blocks))
            token_cost = float(vlm_rules.get("token_cost_per_million", 0.1))
            calc_cost: float = pages * (token_cost / 1000)
            return calc_cost

        return 0.0

    def _log_to_ledger(self, extraction: ExtractedDocument) -> None:
        """Logs an extraction event to the JSONL ledger."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure cost is accurate
        if extraction.cost_estimate == 0.0:
            extraction.cost_estimate = self._calculate_actual_cost(extraction)

        entry = {
            "doc_id": extraction.doc_id,
            "strategy_used": extraction.strategy_used,
            "confidence_score": extraction.confidence_score,
            "cost_estimate": extraction.cost_estimate,
            "processing_time": extraction.extraction_time_seconds,
            "timestamp": datetime.now().isoformat(),
            "escalation_path": extraction.metadata.get("escalation_path", []),
        }

        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _select_strategy(self, profile: DocumentProfile) -> BaseExtractor:
        """Selects the appropriate strategy based on the DocumentProfile."""
        if profile.origin_type == "scanned_image":
            return cast(BaseExtractor, self.vision)
        elif profile.layout_complexity == "single_column":
            return cast(BaseExtractor, self.fast)
        return cast(BaseExtractor, self.layout)

    def route_and_extract(
        self,
        pdf_path: Path,
        profile: DocumentProfile,
        max_pages: Optional[int] = None,
    ) -> ExtractedDocument:
        """Routes the document with PER-PAGE escalation."""
        # 1. Primary Strategy Selection
        primary_strategy = self._select_strategy(profile)

        # 2. Sequential Execution (Master Thinker: Per-Page Evaluation)
        # We start with the primary strategy but check for per-page failures
        extraction = primary_strategy.extract(pdf_path, profile.doc_id, max_pages=max_pages)
        escalation_path = [primary_strategy.__class__.__name__.replace("Extractor", "")]

        t_rules: Dict[str, Any] = self.rules["thresholds"]["triage"]

        # Master Thinker: check for Empty Tables or low per-page confidence
        needs_escalation = False

        # Check Strategy A -> B escalation
        if extraction.strategy_used == "A_FastText":
            if extraction.confidence_score < float(t_rules["min_conf_a"]):
                needs_escalation = True

        # Check Strategy B -> C escalation
        if extraction.strategy_used == "B_Layout" or needs_escalation:
            min_conf_b = float(t_rules["min_conf_b"])
            if needs_escalation or extraction.confidence_score < min_conf_b:
                logger.warning(f"Escalating {profile.doc_id} to Vision (Tier C)")
                extraction = self.vision.extract(pdf_path, profile.doc_id, max_pages=max_pages)
                escalation_path.append("Vision")
            else:
                # Check for "Empty Tables" on layout results
                empty_count = sum(
                    1 for t in extraction.tables if t.caption and "EmptyTable:True" in t.caption
                )
                if empty_count > 0:
                    logger.warning(f"Detected {empty_count} empty tables. Escalating to C.")
                    extraction = self.vision.extract(pdf_path, profile.doc_id, max_pages=max_pages)
                    escalation_path.append("Vision (TableFix)")

        # Finalize and log
        extraction.metadata["escalation_path"] = escalation_path

        thresh_c = float(t_rules["min_conf_c"])
        if extraction.confidence_score < thresh_c:
            extraction.needs_human_review = True
            logger.error(f"Extraction for {profile.doc_id} has low final confidence.")

        self._log_to_ledger(extraction)
        return extraction
