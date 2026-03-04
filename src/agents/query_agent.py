import logging
import os
import sqlite3
from typing import Annotated, Dict, List, Literal, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agents.indexer import PageIndexManager
from src.data.vector_store import VectorStore
from src.models.core import ProvenanceChain, ProvenanceCitation

logger = logging.getLogger(__name__)

# --- State Definition ---


class QueryState(TypedDict):
    """The state of the query agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    doc_id: Optional[str]
    citations: List[ProvenanceCitation]
    final_answer: Optional[str]


# --- Tools ---


@tool  # type: ignore[misc]
async def pageindex_navigate(query: str, doc_id: str) -> str:
    """
    Navigates the hierarchical PageIndex tree to find relevant sections for a query.
    Useful for high-level discovery and locating specific chapters or sections.
    """
    manager = PageIndexManager()
    nodes = await manager.navigate(doc_id, query)

    if not nodes:
        return f"No relevant sections found in PageIndex for '{query}'."

    results = []
    for node in nodes:
        results.append(
            f"Section: {node.title} (Pages {node.page_start}-{node.page_end})\n"
            f"Summary: {node.summary}\n"
        )
    return "\n".join(results)


@tool  # type: ignore[misc]
def semantic_search(query: str, k: int = 5) -> str:
    """
    Performs a semantic vector search across document chunks (LDUs).
    Best for finding specific details, paragraphs, or context-heavy information.
    """
    vs = VectorStore()
    docs = vs.search(query, k=k)

    if not docs:
        return "No relevant chunks found in vector store."

    results = []
    for doc in docs:
        meta = doc.metadata
        results.append(
            f"[Source: {meta.get('doc_id')}, Page: {meta.get('page_refs')}, "
            f"Hash: {meta.get('content_hash')}, BBox: {meta.get('bbox')}]\n"
            f"Content: {doc.page_content}\n"
            f"---"
        )
    return "\n".join(results)


@tool  # type: ignore[misc]
def structured_query(sql_query: str) -> str:
    """
    Executes a SQL query against the FactTable (SQLite).
    Best for numerical analysis, fiscal data comparison, and structured table data queries.
    Only use valid SQLite syntax against the 'facts' table.
    Schema: facts(doc_id, page_number, fact_key, fact_value, unit, confidence, source_chunk_hash)
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
    Multi-tool LangGraph agent for document Q&A with spatial provenance.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0,
            max_tokens=2048,
        )
        self.tools = [pageindex_navigate, semantic_search, structured_query]
        self.model_with_tools = self.llm.bind_tools(self.tools)

        # Build Graph
        builder = StateGraph(QueryState)

        builder.add_node("agent", self._call_model)
        builder.add_node("tools", ToolNode(self.tools))

        builder.set_entry_point("agent")
        builder.add_conditional_edges(
            "agent",
            self._should_continue,
        )
        builder.add_edge("tools", "agent")

        self.graph = builder.compile()

    def _should_continue(self, state: QueryState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    def _call_model(self, state: QueryState) -> Dict[str, List[BaseMessage]]:
        response = self.model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    async def run(self, query: str, doc_id: Optional[str] = None) -> ProvenanceChain:
        """Runs the query agent and returns a structured ProvenanceChain."""
        system_prompt = (
            "You are a professional Document Intelligence Agent. Your goal is to "
            "provide accurate answers based ONLY on the provided document data.\n"
            "For every fact you state, you MUST provide a citation in the format: "
            "[Source: DOC_ID, Page: PAGE, Hash: HASH].\n"
            "If spatial data (BBox) is available in the tool output, keep track of it.\n"
            "Be concise and professional."
        )
        inputs = {
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)],
            "doc_id": doc_id,
            "citations": [],
            "final_answer": None,
        }

        final_state = await self.graph.ainvoke(inputs)
        answer = final_state["messages"][-1].content

        # Simple citation extraction regex-based
        import re

        citation_matches = re.finditer(r"\[Source: (.*?), Page: (.*?), Hash: (.*?)\]", answer)
        citations = []
        for match in citation_matches:
            # In a real implementation, we would look up the full LDU info from a cache
            # or from the tool outputs to fill in the BBox and excerpt.
            # For now, we'll populate what we can.
            citations.append(
                ProvenanceCitation(
                    document_name=match.group(1),
                    page_number=(
                        int(eval(match.group(2))[0])
                        if "[" in match.group(2)
                        else int(match.group(2))
                    ),
                    content_hash=match.group(3),
                    excerpt="Excerpt placeholder",
                )
            )

        return ProvenanceChain(answer_text=answer, citations=citations, is_verified=False)
