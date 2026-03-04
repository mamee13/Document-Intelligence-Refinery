import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.query_agent import QueryAgent
from src.models.core import ProvenanceChain

logger = logging.getLogger(__name__)


class AuditManager:
    """
    Audit Mode: Verifies claims against the document source using the QueryAgent.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0,
        )
        self.query_agent = QueryAgent(model_name=model_name)

    async def verify_claim(self, claim: str, doc_id: Optional[str] = None) -> ProvenanceChain:
        """
        Verifies a specific claim and returns a ProvenanceChain with verification status.
        """
        logger.info(f"Verifying claim: {claim}")

        # 1. Ask the QueryAgent for information related to the claim
        query = f"Verify the following claim using the document data: '{claim}'. Provide specific excerpts and citations."
        provenance = await self.query_agent.run(query, doc_id=doc_id)

        # 2. Use the LLM to determine final verification status based on the agent's findings
        status_prompt = (
            f"Given the claim: '{claim}'\n"
            f"And the evidence found: '{provenance.answer_text}'\n\n"
            "Classify this claim as ONE of the following: 'Verified', 'Contradicted', or 'Unverifiable'. "
            "Return ONLY the status string."
        )

        response = self.llm.invoke(status_prompt)
        status = response.content.strip()

        provenance.is_verified = status == "Verified"

        # Append status to answer text for clarity
        provenance.answer_text = f"Audit Status: {status}\n\n{provenance.answer_text}"

        return provenance
