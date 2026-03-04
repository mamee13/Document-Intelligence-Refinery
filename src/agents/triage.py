from pathlib import Path
from typing import Any, Dict, Optional

import pdfplumber

from src.agents.domain_classifiers import (
    BaseDomainClassifier,
    KeywordDomainClassifier,
)
from src.models.core import DocumentProfile, LayoutComplexity, OriginType
from src.utils.config import RULES


class TriageAgent:
    """
    Analyzes the first N pages of a PDF to empirically determine OriginType.
    Uses thresholds in extraction_rules.yaml for classification.
    """

    def __init__(
        self,
        max_pages_to_analyze: Optional[int] = None,
        domain_classifier: Optional[BaseDomainClassifier] = None,
    ):
        self.max_pages = max_pages_to_analyze
        self.thresholds = RULES["thresholds"]["triage"]
        self.domain_hints = RULES.get("domain_hints", {})
        self.cost_config = RULES.get(
            "cost_tiers",
            {
                "scanned_base": 0.05,
                "native_base": 0.0,
                "complexity_multiplier": 1.5,
            },
        )

        # Inject or use default Strategy for domain classification
        self.domain_classifier = domain_classifier or KeywordDomainClassifier(self.domain_hints)

    def _extract_text_and_metrics(self, pdf_path: Path) -> Dict[str, Any]:
        """Runs pdfplumber over N pages to extract metrics."""
        text_content = ""
        total_chars = 0
        total_area = 0.0
        total_image_area = 0.0

        zero_char_pages = 0
        multi_col_pages = 0
        tables_detected = 0
        pages_processed = 0
        is_form_fillable = False
        font_types: set[str] = set()

        with pdfplumber.open(pdf_path) as pdf:
            # Check for form-fillable or Acroform
            is_form_fillable = bool(pdf.metadata.get("AcroForm") or pdf.doc.catalog.get("AcroForm"))
            pages_to_check = pdf.pages[: self.max_pages] if self.max_pages else pdf.pages
            pages_processed = len(pages_to_check)

            for page in pages_to_check:
                # 1. Text extraction
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"

                # 2. Area calculations (points squared)
                width = page.width or 1
                height = page.height or 1
                page_area = float(width * height)
                total_area += page_area

                char_count = len(page_text.strip())
                total_chars += char_count

                if char_count == 0:
                    zero_char_pages += 1

                # 3. Image area calculation
                img_area = 0.0
                for img in page.images:
                    w = img.get("width", 0)
                    h = img.get("height", 0)
                    img_area += w * h
                total_image_area += img_area

                # 4. Naive Table detection
                tables = page.find_tables()
                if tables:
                    tables_detected += len(tables)

                # 5. Very basic multi-col heuristic:
                words = page.extract_words()
                if words:
                    x_coords = [w["x0"] for w in words]
                    if x_coords:
                        max_x = max(x_coords)
                        if max_x > width * 0.7 and len(words) > 50:
                            multi_col_pages += 1

                # 6. Font Metadata Analysis
                for obj in page.chars:
                    font_name = obj.get("fontname", "unknown")
                    font_types.add(font_name)

        avg_char_density = total_chars / total_area if total_area > 0 else 0
        avg_image_ratio = total_image_area / total_area if total_area > 0 else 0
        zero_char_ratio = zero_char_pages / pages_processed if pages_processed > 0 else 0
        multi_col_ratio = multi_col_pages / pages_processed if pages_processed > 0 else 0

        return {
            "text": text_content.lower(),
            "is_form_fillable": is_form_fillable,
            "metrics": {
                "avg_char_density": avg_char_density,
                "avg_image_area_ratio": avg_image_ratio,
                "zero_char_ratio": zero_char_ratio,
                "multi_col_ratio": multi_col_ratio,
                "tables_detected": tables_detected,
                "pages_processed": pages_processed,
                "embedded_font_count": len(font_types),
            },
        }

    def _estimate_cost(self, origin: OriginType, complexity: LayoutComplexity) -> float:
        """Derives estimated cost tier from configuration."""
        tiers = RULES.get("cost_tiers", {})
        base_cost = tiers.get("base_per_page", 0.001)

        multiplier = 1.0
        if origin == "scanned_image":
            multiplier *= tiers.get("scanned_multiplier", 5.0)

        if complexity == "multi_column":
            multiplier *= tiers.get("complexity_multiplier", 1.5)
        elif complexity == "table_heavy":
            multiplier *= tiers.get("table_heavy_multiplier", 2.0)

        return float(round(base_cost * multiplier, 4))

    def classify_document(self, pdf_path: str | Path) -> DocumentProfile:
        """Analyzes a PDF and returns a structured DocumentProfile."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc_id = path.stem
        analysis = self._extract_text_and_metrics(path)
        m = analysis["metrics"]

        # Handle edge case: Zero pages detected or empty file
        if m["pages_processed"] == 0:
            return DocumentProfile(
                doc_id=doc_id,
                origin_type="scanned_image",  # Default to safest
                layout_complexity="single_column",
                confidence_score=0.1,
            )

        # 1. Determine Origin Type
        t = self.thresholds
        is_scanned = (
            m["avg_char_density"] <= t["scanned_density_max"]
            or m["zero_char_ratio"] >= t["scanned_zero_char_ratio"]
            or m["avg_image_area_ratio"] >= t["scanned_image_area_ratio_min"]
        )

        origin_type: OriginType = "scanned_image" if is_scanned else "native_digital"

        # 2. Determine Layout Complexity
        layout_complexity: LayoutComplexity = "single_column"
        if not is_scanned:
            if m["tables_detected"] > 0:
                layout_complexity = "table_heavy"
            elif m["multi_col_ratio"] >= t["simple_multi_col_ratio_max"]:
                layout_complexity = "multi_column"
            elif m["avg_char_density"] <= t["native_density_min"]:
                layout_complexity = "multi_column"

        # 2.1 Calculate Confidence Score (Mastered requirement)
        # Higher confidence if signals agree (e.g., low density AND high image ratio)
        conf = 0.95
        if is_scanned:
            # If it's scanned but has many fonts, it might be an OCRed PDF
            # or a complex digital one, lowering confidence in "scanned" label.
            if m["embedded_font_count"] > 2:
                conf -= 0.2
            if m["avg_char_density"] > t["scanned_density_max"] * 0.8:
                conf -= 0.1
        else:
            # Native digital with zero embedded fonts is rare/suspicious
            if m["embedded_font_count"] == 0:
                conf -= 0.3
            if m["zero_char_ratio"] > 0.05:
                conf -= 0.1

        confidence = max(0.1, min(conf, 1.0))

        # 3. Determine Domain Hint using Strategy
        domain = self.domain_classifier.classify(analysis["text"])

        # 4. Estimate Extraction Cost (Master status)
        cost_estimate = self._estimate_cost(origin_type, layout_complexity)

        # Create Profile
        profile_path = Path(".refinery/profiles") / f"{doc_id}.json"

        profile = DocumentProfile(
            doc_id=doc_id,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            domain_hint=domain,
            estimated_extraction_cost=cost_estimate,
            confidence_score=confidence,
        )

        # Save to disk
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            f.write(profile.model_dump_json(indent=2))

        return profile
