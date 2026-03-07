import asyncio
import logging
import sys
from typing import Optional

from src.agents.audit import AuditManager
from src.agents.indexer import PageIndexManager
from src.agents.query_agent import QueryAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_query(query: str, doc_id: Optional[str] = None) -> None:
    agent = QueryAgent()
    result = await agent.run(query, doc_id=doc_id)
    print(f"\n{'='*20} QUERY RESULT {'='*20}")
    print(f"ANSWER: {result.answer_text}")
    print(f"\nCITATIONS ({len(result.citations)}):")
    for cit in result.citations:
        print(f" - [Page {cit.page_number}] " f"{cit.document_name} ({cit.content_hash[:8]})")
    if result.chain_hash:
        print(f"\nPROVENANCE HASH: {result.chain_hash}")
    print(f"{'='*54}\n")


async def run_audit(claim: str, doc_id: Optional[str] = None) -> None:
    manager = AuditManager()
    result = await manager.verify_claim(claim, doc_id=doc_id)
    print(f"\n{'='*20} AUDIT VERIFICATION {'='*20}")
    print(result.answer_text)
    print(f"{'='*54}\n")


async def run_navigate(doc_id: str, query: str) -> None:
    manager = PageIndexManager()
    nodes = await manager.navigate(doc_id, query)
    print(f"\n{'='*20} PAGEINDEX NAVIGATION {'='*20}")
    if not nodes:
        print("No matches found.")
    else:
        for node in nodes:
            print(f"Section: {node.title} (Pages {node.page_start}-{node.page_end})")
            print(f"Summary: {node.summary}")
            print("-" * 10)
    print(f"{'='*54}\n")


def print_usage() -> None:
    print("Usage: uv run python refinery_cli.py [mode] [args...]")
    print("Modes:")
    print('  query "Your question" [doc_id]')
    print('  audit "Your claim" [doc_id]')
    print('  navigate [doc_id] "Section query"')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "query":
        q = sys.argv[2] if len(sys.argv) > 2 else "What are the key findings?"
        doc_id = sys.argv[3] if len(sys.argv) > 3 else None
        asyncio.run(run_query(q, doc_id))
    elif mode == "audit":
        c = sys.argv[2] if len(sys.argv) > 2 else "Revenue was over 10 billion."
        doc_id = sys.argv[3] if len(sys.argv) > 3 else None
        asyncio.run(run_audit(c, doc_id))
    elif mode == "navigate":
        if len(sys.argv) < 4:
            print("Error: navigate requires [doc_id] [query]")
            sys.exit(1)
        doc_id = sys.argv[2]
        query = sys.argv[3]
        asyncio.run(run_navigate(doc_id, query))
    else:
        print_usage()
