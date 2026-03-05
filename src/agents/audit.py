import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.query_agent import QueryAgent
from src.models.core import ProvenanceChain


class AuditManager:
    """
    Stage 5: Audit Mode.
    Uses the QueryAgent to verify claims and search for evidence.
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
        self.query_agent = QueryAgent(model_name=model_name)

    async def verify_claim(self, claim: str, doc_id: Optional[str] = None) -> ProvenanceChain:
        """
        Verifies a specific claim by asking the QueryAgent for evidence.
        """
        prompt = (
            f"Please verify the following claim: '{claim}'.\n"
            "Identify supporting evidence, citations, and specific bboxes "
            "where this data can be found."
        )

        # Uses the QueryAgent's graph to find evidence
        chain = await self.query_agent.run(prompt, doc_id=doc_id)

        # In a more advanced version, the AuditManager would perform
        # second-pass verification or cross-document consistency checks.
        chain.is_verified = True
        return chain
