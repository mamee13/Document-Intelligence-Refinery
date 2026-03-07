import logging
import re
from typing import Any, Dict, List

from src.models.core import LDU

logger = logging.getLogger(__name__)


class ChunkValidator:
    """
    Programmatically enforces the 5 core chunking rules before LDUs are emitted.
    """

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules

    def validate(self, ldus: List[LDU]) -> List[LDU]:
        """
        Runs all validation rules and returns the (possibly modified) LDUs.
        """
        if not ldus:
            return ldus

        self._validate_table_integrity(ldus)
        self._validate_figure_captions(ldus)
        self._validate_list_integrity(ldus)
        self._validate_section_propagation(ldus)
        self._resolve_cross_references(ldus)

        return ldus

    def _validate_table_integrity(self, ldus: List[LDU]) -> None:
        """Rule 1: Tables must be kept intact with headers."""
        for ldu in ldus:
            if ldu.chunk_type == "table":
                # Check if it looks like a markdown table
                if "|" not in ldu.content or "---" not in ldu.content:
                    logger.warning(
                        f"LDU {ldu.chunk_id} marked as table but missing markdown structure."
                    )

                # Check if caption is present if it was expected
                if "Table Caption:" not in ldu.content and ldu.content.startswith("|"):
                    # This is a soft check as not all tables have captions
                    pass

    def _validate_figure_captions(self, ldus: List[LDU]) -> None:
        """Rule 2: Figure captions must be attached to figures."""
        for ldu in ldus:
            if ldu.chunk_type == "figure":
                if "Figure Caption:" not in ldu.content:
                    logger.error(f"Rule Violation: Figure {ldu.chunk_id} missing caption prefix.")

    def _validate_list_integrity(self, ldus: List[LDU]) -> None:
        """Rule 3: Numbered/bulleted lists must be preserved as single units."""
        for ldu in ldus:
            if ldu.chunk_type == "list":
                # Ensure it contains list markers
                if not re.search(r"^(\d+[\.\)]|[\u2022\-\*])", ldu.content, re.MULTILINE):
                    logger.warning(f"LDU {ldu.chunk_id} marked as list but missing markers.")

    def _validate_section_propagation(self, ldus: List[LDU]) -> None:
        """Rule 4: Section headers must be propagated as parent metadata."""
        current_header = None
        for ldu in ldus:
            if ldu.chunk_type == "section_header":
                current_header = ldu.content
            else:
                if ldu.parent_section and ldu.parent_section != current_header:
                    # This happens if chunker lost track or we have nested headers
                    # For now we just log it
                    logger.debug(
                        f"Mismatched parent_section for {ldu.chunk_id}: {ldu.parent_section} vs {current_header}"
                    )

    def _resolve_cross_references(self, ldus: List[LDU]) -> None:
        """Rule 5: Cross-references must be resolved and stored as relationships."""
        # Build a mapping of common reference names to chunk IDs
        # e.g., "Table 1" -> "table_abc123"
        ref_map = {}
        for ldu in ldus:
            if ldu.chunk_type == "table":
                match = re.search(r"Table (\d+)", ldu.content, re.IGNORECASE)
                if match:
                    ref_map[match.group(0).lower()] = ldu.chunk_id
            elif ldu.chunk_type == "figure":
                match = re.search(r"Figure (\d+)", ldu.content, re.IGNORECASE)
                if match:
                    ref_map[match.group(0).lower()] = ldu.chunk_id

        for ldu in ldus:
            resolved_refs = []
            for raw_ref in ldu.cross_references:
                clean_ref = raw_ref.lower()
                if clean_ref in ref_map:
                    resolved_refs.append(ref_map[clean_ref])
                else:
                    resolved_refs.append(raw_ref)  # Keep raw if not found

            ldu.cross_references = resolved_refs
