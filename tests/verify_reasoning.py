import asyncio
import logging

from src.agents.audit import AuditManager
from src.agents.query_agent import QueryAgent

logging.basicConfig(level=logging.INFO)


async def run_integration_test() -> None:
    print("🚀 Starting Reasoning & Provenance Integration Test...")

    # 1. Test Query Agent
    print("\n[Step 1] Testing QueryAgent...")
    agent = QueryAgent()
    query = "What is the primary objective of the Consumer Price Index June 2025 document?"
    result = await agent.run(query)
    print(f"Query Result answer length: {len(result.answer_text)}")
    print(f"Number of citations: {len(result.citations)}")

    # 2. Test Audit Manager
    print("\n[Step 2] Testing AuditManager...")
    audit = AuditManager()
    claim = "The June 2025 CPI index value is 187.5."
    audit_result = await audit.verify_claim(claim)
    print(f"Audit Result Status: {audit_result.answer_text.splitlines()[0]}")
    print(f"Number of audit citations: {len(audit_result.citations)}")

    print("\n✅ Reasoning & Provenance integration test script completed.")


if __name__ == "__main__":
    asyncio.run(run_integration_test())
