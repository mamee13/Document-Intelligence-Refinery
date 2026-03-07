import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import httpx
from dotenv import load_dotenv

from src.models.core import LDU, PageIndex, PageIndexNode
from src.utils.config import RULES

if TYPE_CHECKING:
    from src.data.vector_store import VectorStore

load_dotenv()
logger = logging.getLogger(__name__)


class PageIndexManager:
    """
    Stage 4: Builds a hierarchical PageIndex from LDUs and generates summaries.
    Upgraded for nesting and entity extraction.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.vlm_config = RULES.get("vlm", {})
        self.model = self.vlm_config.get("model_name", "google/gemini-2.0-flash-001")

    async def _summarize_section(self, title: str, content: str) -> str:
        """Generates a 2-3 sentence summary of the section using LLM."""
        if not self.api_key:
            return "[MOCK SUMMARY: No API Key]"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a professional document analyst. Provide a concise 2-3 sentence summary of the following section.",
                            },
                            {
                                "role": "user",
                                "content": f"Section Title: {title}\n\nContent:\n{content[:5000]}",
                            },
                        ],
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as e:
            return f"[Error summarizing section: {str(e)}]"

    async def _extract_entities(self, title: str, content: str) -> List[str]:
        """Extracts key entities (orgs, dates, amounts) from section content."""
        if not self.api_key or not content.strip():
            return []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Extract 3-5 key entities (organizations, locations, specific dates, or financial amounts) from the text. Return as a comma-separated list.",
                            },
                            {
                                "role": "user",
                                "content": f"Text:\n{content[:3000]}",
                            },
                        ],
                    },
                    timeout=20.0,
                )
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"]
                entities = [e.strip() for e in text.split(",") if e.strip()]
                return entities[:10]
        except Exception:
            return []

    def _build_hierarchy(self, ldus: List[LDU]) -> List[PageIndexNode]:
        """
        Groups LDUs into sections based on section_header chunks.
        Enforces true hierarchical nesting based on header patterns.
        """
        root_nodes: List[PageIndexNode] = []
        stack: List[PageIndexNode] = []

        for ldu in ldus:
            if ldu.chunk_type == "section_header":
                # Infer level: Count dots in numeric prefixes (e.g., 1.2.3 -> level 3)
                # or look for indentation/case if available (heuristic)
                level = 1
                match = re.search(r"^([\d\.]+)", ldu.content)
                if match:
                    # e.g., "4.1.2" -> dots = 2 -> level 3
                    level = 1 + match.group(1).count(".")
                    if (
                        ldu.content.startswith(match.group(1))
                        and ldu.content[len(match.group(1)) :].strip() == ""
                    ):
                        # Just a number, might be a page number mistakenly caught, keep level 1
                        pass

                node = PageIndexNode(
                    section_id=ldu.chunk_id,
                    title=ldu.content,
                    page_start=ldu.page_refs[0],
                    page_end=ldu.page_refs[-1],
                    level=level,
                    children=[],
                )

                # Maintain stack for nesting
                while stack and stack[-1].level >= level:
                    stack.pop()

                if stack:
                    stack[-1].children.append(node)
                else:
                    root_nodes.append(node)

                stack.append(node)
            else:
                if stack:
                    active_node = stack[-1]
                    active_node.page_end = max(active_node.page_end, ldu.page_refs[-1])
                    if ldu.chunk_type not in active_node.data_types_present:
                        active_node.data_types_present.append(ldu.chunk_type)

        return root_nodes

    async def _process_node_recursive(self, node: PageIndexNode, ldus: List[LDU]) -> None:
        """Recursively populates summaries and entities for a node and its children."""
        # 1. Robust content matching with multiple strategies
        clean_title = node.title.lower().strip()

        # Strategy 1: Exact match (case-insensitive)
        section_ldus = [
            chunk
            for chunk in ldus
            if chunk.parent_section and chunk.parent_section.lower().strip() == clean_title
        ]

        # Strategy 2: If no exact match, try partial match (for truncated headers)
        if not section_ldus:
            section_ldus = [
                chunk
                for chunk in ldus
                if chunk.parent_section
                and (
                    clean_title in chunk.parent_section.lower().strip()
                    or chunk.parent_section.lower().strip() in clean_title
                )
            ]

        # Strategy 3: If still no match, use page range overlap
        if not section_ldus:
            section_ldus = [
                chunk
                for chunk in ldus
                if chunk.chunk_type != "section_header"
                and any(p >= node.page_start and p <= node.page_end for p in chunk.page_refs)
            ]

        content_text = "\n".join([l.content for l in section_ldus])

        if content_text.strip():
            node.summary = await self._summarize_section(node.title, content_text)
            node.key_entities = await self._extract_entities(node.title, content_text)
        else:
            node.summary = "No content directly linked to this section header."

        # Process children
        for child in node.children:
            await self._process_node_recursive(child, ldus)

    async def create_index(self, doc_id: str, ldus: List[LDU]) -> PageIndex:
        """
        Builds the hierarchy and populates summaries/entities.
        """
        root_nodes = self._build_hierarchy(ldus)

        # Populate data for each node in the tree
        for node in root_nodes:
            await self._process_node_recursive(node, ldus)

        index = PageIndex(doc_id=doc_id, root_nodes=root_nodes)

        # Save to disk
        output_path = Path(".refinery/pageindex") / f"{doc_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(index.model_dump_json(indent=2))

        return index

    async def navigate(
        self,
        doc_id: str,
        query: str,
        k: int = 3,
        vector_store: Optional["VectorStore"] = None,
    ) -> List[PageIndexNode]:
        """
        Traverses the index using semantic similarity against section summaries.
        Returns top-k most relevant sections.
        """
        # Load the index
        index_path = Path(".refinery/pageindex") / f"{doc_id}.json"
        if not index_path.exists():
            return []

        with open(index_path, "r") as f:
            data = json.load(f)
            index = PageIndex.model_validate(data)

        from src.data.vector_store import VectorStore

        vs = vector_store or VectorStore()

        # Flatten tree for searching
        all_nodes: List[PageIndexNode] = []

        def _flatten(nodes: List[PageIndexNode]) -> None:
            for n in nodes:
                all_nodes.append(n)
                _flatten(n.children)

        _flatten(index.root_nodes)

        # Search
        results = vs.search(query, k=k * 3)
        relevant_titles = set()
        for r in results:
            if r.metadata.get("chunk_type") == "section_header":
                relevant_titles.add(r.page_content.lower().strip())
            elif r.metadata.get("parent_section"):
                relevant_titles.add(r.metadata.get("parent_section").lower().strip())

        # Filter and maintain order
        matched = []
        for node in all_nodes:
            if node.title.lower().strip() in relevant_titles:
                matched.append(node)

        return matched[:k]
