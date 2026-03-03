from pathlib import Path
from typing import Any, Dict, Optional

import pdfplumber

from src.models.core import DocumentProfile, LayoutComplexity, OriginType
from src.utils.config import RULES


class TriageAgent:
    """
    Analyzes the first N pages of a PDF to empirically determine its OriginType
    and LayoutComplexity based on thresholds in extraction_rules.yaml.
    """

    def __init__(self, max_pages_to_analyze: int = 10):
        self.max_pages = max_pages_to_analyze
        self.thresholds = RULES["thresholds"]["triage"]
        self.domain_hints = RULES["domain_hints"]

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

        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = pdf.pages[: self.max_pages]
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
                    # Look for horizontal gaps indicating columns
                    x_coords = [w["x0"] for w in words]
                    if x_coords:
                        max_x = max(x_coords)
                        # Multi-col proxy: wide span but low middle density
                        if max_x > width * 0.7 and len(words) > 50:
                            multi_col_pages += 1

        # Calculate final metrics
        avg_char_density = total_chars / total_area if total_area > 0 else 0
        avg_image_ratio = total_image_area / total_area if total_area > 0 else 0
        zero_char_ratio = zero_char_pages / pages_processed if pages_processed > 0 else 0
        multi_col_ratio = multi_col_pages / pages_processed if pages_processed > 0 else 0

        return {
            "text": text_content.lower(),
            "metrics": {
                "avg_char_density": avg_char_density,
                "avg_image_area_ratio": avg_image_ratio,
                "zero_char_ratio": zero_char_ratio,
                "multi_col_ratio": multi_col_ratio,
                "tables_detected": tables_detected,
            },
        }

    def _determine_domain(self, text: str) -> Optional[str]:
        """Simple keyword matching to guess the domain."""
        for domain, keywords in self.domain_hints.items():
            for kw in keywords:
                if kw in text:
                    return str(domain)
        return None

    def classify_document(self, pdf_path: str | Path) -> DocumentProfile:
        """Analyzes a PDF and returns a structured DocumentProfile."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc_id = path.stem
        analysis = self._extract_text_and_metrics(path)
        m = analysis["metrics"]

        # 1. Determine Origin Type (Scanned vs Native)
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
                # If it's native but has low density, it's likely complex
                layout_complexity = "multi_column"

        # 3. Determine Domain Hint
        domain = self._determine_domain(analysis["text"])

        # Create Profile
        profile_path = Path(".refinery/profiles") / f"{doc_id}.json"

        profile = DocumentProfile(
            doc_id=doc_id,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            domain_hint=domain,
            estimated_extraction_cost=0.0,
        )

        # Save to disk
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            f.write(profile.model_dump_json(indent=2))

        return profile
