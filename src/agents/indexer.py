import json
import logging
import os
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

    def _build_hierarchy(self, ldus: List[LDU]) -> List[PageIndexNode]:
        """
        Groups LDUs into sections based on section_header chunks.
        Currently builds a flat list of nodes, but could be extended for nesting.
        """
        nodes: List[PageIndexNode] = []
        current_node: Optional[PageIndexNode] = None
        current_content: List[str] = []

        for ldu in ldus:
            if ldu.chunk_type == "section_header":
                # Finalize previous node
                if current_node:
                    current_node.summary = "Summarizing..."  # Placeholder for async step
                    nodes.append(current_node)

                # Start new node
                current_node = PageIndexNode(
                    section_id=ldu.chunk_id,
                    title=ldu.content,
                    page_start=ldu.page_refs[0],
                    page_end=ldu.page_refs[-1],
                    level=1,  # Default level
                )
                current_content = []
            else:
                if current_node:
                    current_node.page_end = max(current_node.page_end, ldu.page_refs[-1])
                    current_content.append(ldu.content)
                    if ldu.chunk_type not in current_node.data_types_present:
                        current_node.data_types_present.append(ldu.chunk_type)

        if current_node:
            nodes.append(current_node)

        return nodes

    async def create_index(self, doc_id: str, ldus: List[LDU]) -> PageIndex:
        """
        Builds the hierarchy and populates summaries.
        """
        nodes = self._build_hierarchy(ldus)

        # Generate summaries (sequentially for now to avoid rate limits,
        # but could be parallelized)
        for node in nodes:
            # Reconstruct content for summary
            # We filter LDUs for this section
            section_ldus = [l for l in ldus if l.parent_section == node.title]
            content_text = "\n".join([l.content for l in section_ldus])

            # If no content was linked via parent_section, try using block order
            # (already done in _build_hierarchy logic if we kept it)
            # For now, let's just use the section_ldus
            if content_text:
                node.summary = await self._summarize_section(node.title, content_text)
            else:
                node.summary = "No content found for this section."

        index = PageIndex(doc_id=doc_id, root_nodes=nodes)

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

        # Use VectorStore's embedding logic to find relevant sections
        from src.data.vector_store import VectorStore

        vs = vector_store or VectorStore()
        # Collect all nodes
        all_nodes = index.root_nodes  # Simple flat search for now

        # We'll use a simple approach: embed the query and compare with
        # (embedded) summaries. For now, since we haven't stored node embeddings,
        # we'll use a simpler semantic search over the summary text.
        # Efficient way: The summary is already stored in our vector store as LDUs,
        # but the PageIndexNode summary is a higher-level abstraction.

        # Let's use the VectorStore to find the best LDUs, then resolve them to nodes.
        results = vs.search(query, k=k * 2)
        relevant_node_titles = set()
        for r in results:
            if r.metadata.get("chunk_type") == "section_header":
                relevant_node_titles.add(r.page_content)  # Header content is the title
            elif r.metadata.get("parent_section"):
                relevant_node_titles.add(r.metadata.get("parent_section"))

        # Filter and sort nodes by relevance
        matched_nodes = [node for node in all_nodes if node.title in relevant_node_titles]
        return matched_nodes[:k]
