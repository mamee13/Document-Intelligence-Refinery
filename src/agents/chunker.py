import hashlib
import re
from typing import List, Optional

import tiktoken

from src.models.core import LDU, ExtractedDocument
from src.utils.config import RULES


class ChunkingEngine:
    """
    Stage 3: Transforms ExtractedDocument into LDUs.
    Enforces rules for table integrity and section hierarchy.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.rules = RULES.get("chunking_rules", {})

    def _generate_id(self, content: str) -> str:
        """Generates a stable ID based on the content hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a string."""
        return len(self.encoder.encode(text))

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        """
        Processes an ExtractedDocument into LDUs.
        """
        ldus: List[LDU] = []
        current_section: Optional[str] = None

        # 1. Process Tables (Rule 1: Table Integrity)
        # We store the entire table as a single LDU.
        for table in doc.tables:
            content = table.markdown_grid
            if table.caption:
                content = f"Table Caption: {table.caption}\n\n{content}"

            ldu = LDU(
                chunk_id=f"table_{self._generate_id(content)[:8]}",
                doc_id=doc.doc_id,
                content=content,
                chunk_type="table",
                page_refs=[table.page_number],
                bounding_box=table.bbox,
                parent_section=current_section,
                token_count=self._count_tokens(content),
                content_hash=self._generate_id(content),
            )
            ldus.append(ldu)

        # 2. Process Figures (Rule 2: Figure Captions)
        for figure in doc.figures:
            content = f"Figure Caption: {figure.caption or 'None'}"
            ldu = LDU(
                chunk_id=f"fig_{self._generate_id(content)[:8]}",
                doc_id=doc.doc_id,
                content=content,
                chunk_type="figure",
                page_refs=[figure.page_number],
                bounding_box=figure.bbox,
                parent_section=current_section,
                token_count=self._count_tokens(content),
                content_hash=self._generate_id(content),
            )
            ldus.append(ldu)

        # 3. Process Text Blocks (Rule 3 & 4: Lists and Sections)
        # We'll use a simple heuristic for headers and group lists.
        # LayoutExtractor (Docling) gives us hints, for Strategy A we infer.

        i = 0
        while i < len(doc.text_blocks):
            block = doc.text_blocks[i]
            text = block.text.strip()

            if not text:
                i += 1
                continue

            # Heuristic for Lists (Rule 3)
            # Group consecutive blocks starting with bullets or numbers
            is_list_item = re.match(r"^(\d+[\.\)]|[\u2022\-\*])", text)
            if is_list_item:
                list_items = [text]
                start_page = block.page_number
                end_page = block.page_number
                combined_bbox = block.bbox

                j = i + 1
                while j < len(doc.text_blocks):
                    next_block = doc.text_blocks[j]
                    next_text = next_block.text.strip()
                    if re.match(r"^(\d+[\.\)]|[\u2022\-\*])", next_text):
                        list_items.append(next_text)
                        end_page = next_block.page_number
                        # Update combined_bbox (simplified)
                        j += 1
                    else:
                        break

                content = "\n".join(list_items)
                ldu = LDU(
                    chunk_id=f"list_{self._generate_id(content)[:8]}",
                    doc_id=doc.doc_id,
                    content=content,
                    chunk_type="list",
                    page_refs=list(range(start_page, end_page + 1)),
                    bounding_box=combined_bbox,
                    parent_section=current_section,
                    token_count=self._count_tokens(content),
                    content_hash=self._generate_id(content),
                )
                ldus.append(ldu)
                i = j
                continue

            # Heuristic for Section Header (Rule 4)
            # Short line, maybe title case or ending with no period
            # We already ruled out lists above.
            is_header = len(text) < 80 and not text.endswith(".") and any(c.isupper() for c in text)

            if is_header:
                current_section = text
                ldu = LDU(
                    chunk_id=f"head_{self._generate_id(text)[:8]}",
                    doc_id=doc.doc_id,
                    content=text,
                    chunk_type="section_header",
                    page_refs=[block.page_number],
                    bounding_box=block.bbox,
                    parent_section=None,  # Top-level or self-parented
                    token_count=self._count_tokens(text),
                    content_hash=self._generate_id(text),
                )
                ldus.append(ldu)
                i += 1
                continue

            # Default Paragraph
            max_tokens = self.rules.get("max_tokens", 1024)
            token_count = self._count_tokens(text)
            if token_count > max_tokens:
                # Simple split (Mastered: use params)
                words = text.split()
                # Very basic splitting for now
                text = " ".join(words[:200])
                token_count = self._count_tokens(text)

            ldu = LDU(
                chunk_id=f"para_{self._generate_id(text)[:8]}",
                doc_id=doc.doc_id,
                content=text,
                chunk_type="paragraph",
                page_refs=[block.page_number],
                bounding_box=block.bbox,
                parent_section=current_section,
                token_count=token_count,
                content_hash=self._generate_id(text),
            )
            ldus.append(ldu)
            i += 1

        # 4. Basic Cross-Reference Resolution (Rule 5)
        # Search for "Table X" or "Figure Y" in chunks
        for ldu in ldus:
            # Look for references to other chunks (very basic regex)
            # e.g., (see Table 1)
            res = r"(Table \d+|Figure \d+|Section [A-Z\d\.]+)"
            refs = re.findall(res, ldu.content, re.IGNORECASE)
            if refs:
                # Log strings as potential cross-references
                ldu.cross_references = list(set(refs))

        return ldus
