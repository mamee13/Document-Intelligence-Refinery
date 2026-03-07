import hashlib
import logging
import os
import re
import sqlite3
from typing import (
    Annotated,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
)

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agents.indexer import PageIndexManager
from src.data.vector_store import VectorStore
from src.models.core import BBox, ProvenanceChain, ProvenanceCitation

logger = logging.getLogger(__name__)

# --- State Definition ---


class QueryState(TypedDict):
    """The state of the query agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    doc_id: Optional[str]
    citations: List[ProvenanceCitation]
    final_answer: Optional[str]
    routing_hint: Optional[str]


# --- Tools ---


@tool  # type: ignore[misc]
async def pageindex_navigate(query: str, doc_id: str) -> str:
    """
    Navigates the hierarchical PageIndex tree to find relevant sections.
    Useful for high-level discovery and locating specific chapters.
    """
    manager = PageIndexManager()
    nodes = await manager.navigate(doc_id, query)

    if not nodes:
        return f"No relevant sections found in PageIndex for '{query}'."

    results = []
    for node in nodes:
        results.append(
            f"Section: {node.title} (Pages {node.page_start}-"
            f"{node.page_end})\nSummary: {node.summary}\n"
        )
    return "\n".join(results)


@tool  # type: ignore[misc]
def semantic_search(query: str, k: int = 5) -> str:
    """
    Performs a semantic vector search across document chunks (LDUs).
    Best for finding specific details, paragraphs, or context-heavy info.
    """
    vs = VectorStore()
    docs = vs.search(query, k=k)

    if not docs:
        return "No relevant chunks found in vector store."

    results = []
    for doc in docs:
        meta = doc.metadata
        bbox_data = meta.get("bbox")
        bbox_str = "None"
        if bbox_data and isinstance(bbox_data, dict):
            # Ensure coordinates are floats and present
            bx0 = bbox_data.get("x0", 0.0)
            by0 = bbox_data.get("y0", 0.0)
            bx1 = bbox_data.get("x1", 0.0)
            by1 = bbox_data.get("y1", 0.0)
            bbox_str = f"[{bx0:.1f}, {by0:.1f}, {bx1:.1f}, {by1:.1f}]"

        results.append(
            f"[Source: {meta.get('doc_id')}, "
            f"Page: {meta.get('page_refs')}, "
            f"Hash: {meta.get('content_hash')}, "
            f"BBox: {bbox_str}]\n"
            f"Content: {doc.page_content}\n"
            f"---"
        )
    return "\n".join(results)


@tool  # type: ignore[misc]
def structured_query(sql_query: str) -> str:
    """
    Executes a SQL query against the FactTable (SQLite).
    Best for numerical analysis and structured table data queries.
    Only use valid SQLite syntax against the 'facts' table.
    Schema: facts(doc_id, page_number, fact_key, fact_value, unit,
    confidence, source_chunk_hash)
    """
    try:
        conn = sqlite3.connect(".refinery/refinery_facts.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        rows = cursor.execute(sql_query).fetchall()
        conn.close()

        if not rows:
            return "No records found matching the SQL query."

        return str([dict(r) for r in rows])
    except Exception as e:
        return f"Error executing SQL: {str(e)}"


# --- Agent Implementation ---


class QueryAgent:
    """
    Multi-tool LangGraph agent with EXPLICIT routing and spatial provenance.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0,
            max_tokens=1024,
        )
        self.tools = [pageindex_navigate, semantic_search, structured_query]
        self.model_with_tools = self.llm.bind_tools(self.tools)

        # Build Graph
        builder = StateGraph(QueryState)

        builder.add_node("router", self._route_node)
        builder.add_node("agent", self._call_model)
        builder.add_node("tools", ToolNode(self.tools))

        builder.set_entry_point("router")
        builder.add_edge("router", "agent")
        builder.add_conditional_edges("agent", self._should_continue)
        builder.add_edge("tools", "agent")

        self.graph = builder.compile()

    def _route_node(self, state: QueryState) -> Dict[str, str]:
        """EXPLICIT routing based on query characteristics."""
        query = state["messages"][-1].content.lower()
        hint = "Use semantic_search for details."

        if any(w in query for w in ["total", "sum", "average", "table", "sql", "count"]):
            hint = "Query involves numerical/tabular data. Prefer structured_query."
        elif any(w in query for w in ["chapter", "section", "index", "overview", "sections"]):
            hint = "Query involves document structure. Prefer pageindex_navigate."

        return {"routing_hint": hint}

    def _should_continue(self, state: QueryState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    def _call_model(self, state: QueryState) -> Dict[str, List[BaseMessage]]:
        # Inject routing hint into the prompt
        messages = list(state["messages"])
        if state.get("routing_hint"):
            messages.insert(0, SystemMessage(content=f"ROUTING HINT: {state['routing_hint']}"))

        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}

    async def run(self, query: str, doc_id: Optional[str] = None) -> ProvenanceChain:
        """Runs the query agent and returns a structured ProvenanceChain."""
        system_prompt = (
            "You are a professional Document Intelligence Agent. Your goal is "
            "to provide accurate answers based ONLY on the document data.\n"
            "For every fact you state, you MUST provide a structured citation "
            "STRICTLY in the following format at the end of your sentence "
            "or paragraph: [Source: DOC_ID, Page: PAGE, Hash: HASH, "
            "BBox: BBOX_STRING].\n"
            "BBOX_STRING MUST be formatted as [x0, y0, x1, y1] or 'None'.\n"
            "Be concise and professional."
        )
        inputs = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ],
            "doc_id": doc_id,
            "citations": [],
            "final_answer": None,
        }

        final_state = await self.graph.ainvoke(inputs)
        answer = final_state["messages"][-1].content

        # Robust citation extraction
        patterns = [
            r"\[Source:\s*(.*?),\s*Page:\s*(.*?),\s*Hash:\s*(.*?),\s*" r"BBox:\s*(.*?)\]",
            r"Source:\s*(.*?),\s*Page:\s*(.*?),\s*Hash:\s*(.*?),\s*" r"BBox:\s*(\[.*?\]|None)",
        ]
        citations = []
        for pattern in patterns:
            citation_matches = re.finditer(pattern, answer)
            for match in citation_matches:
                bbox_raw = match.group(4)
                bbox_obj = None
                if bbox_raw and "None" not in bbox_raw and "[" in bbox_raw:
                    try:
                        nums = re.findall(r"[-+]?\d*\.\d+|\d+", bbox_raw)
                        if len(nums) >= 4:
                            bbox_obj = BBox(
                                x0=float(nums[0]),
                                y0=float(nums[1]),
                                x1=float(nums[2]),
                                y1=float(nums[3]),
                            )
                    except Exception as e:
                        logger.debug(f"Failed to parse bbox: {e}")

                # Extract page number properly
                page_str = match.group(2)
                try:
                    p_nums = re.findall(r"\d+", page_str)
                    page_num = int(p_nums[0]) if p_nums else 1
                except Exception:
                    page_num = 1

                citations.append(
                    ProvenanceCitation(
                        document_name=match.group(1).strip(),
                        page_number=page_num,
                        content_hash=match.group(3).strip(),
                        bbox=bbox_obj,
                        excerpt="Refer to source content.",
                    )
                )

        # Deduplicate citations by hash
        seen_hashes = set()
        unique_citations = []
        for c in citations:
            if c.content_hash not in seen_hashes:
                unique_citations.append(c)
                seen_hashes.add(c.content_hash)
        citations = unique_citations

        # Aggregate chain-level provenance summaries
        chain_hash = None
        if citations:
            hashes = [c.content_hash for c in citations if c.content_hash]
            hash_input = "".join(sorted(hashes))
            if hash_input:
                chain_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        return ProvenanceChain(
            answer_text=answer,
            citations=citations,
            chain_hash=chain_hash,
            is_verified=False,
        )
