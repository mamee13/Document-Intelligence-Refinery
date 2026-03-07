import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
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
        Verifies a specific claim by asking the QueryAgent for evidence
        and then performing a cross-check against the raw source content.
        """
        prompt = (
            f"Please verify the following claim: '{claim}'.\n"
            "Identify supporting evidence, citations, and specific bboxes "
            "where this data can be found."
        )

        # 1. Get initial evidence from QueryAgent
        chain = await self.query_agent.run(prompt, doc_id=doc_id)

        if not chain.citations:
            chain.is_verified = False
            chain.answer_text = f"UNVERIFIABLE: No evidence found for claim: {claim}"
            return chain

        # 2. Second-pass verification: Check raw content
        from src.data.vector_store import VectorStore

        vs = VectorStore()

        verification_results = []
        for citation in chain.citations:
            doc = vs.get_by_hash(citation.content_hash)
            if doc:
                # Ask LLM if this specific chunk supports the claim
                check_response = await self.llm.ainvoke(
                    [
                        SystemMessage(
                            content="You are an Auditor. Verify if the given text snippet supports the claim. Answer only with 'VERIFIED', 'CONTRADICTED', or 'NEUTRAL'."
                        ),
                        HumanMessage(
                            content=f"Claim: {claim}\n\nEvidence Link: {doc.page_content}"
                        ),
                    ]
                )
                result = check_response.content.strip().upper()
                verification_results.append(result)
                citation.excerpt = doc.page_content[:200] + "..."  # Populate excerpt
            else:
                verification_results.append("NOT_FOUND")

        # 3. Aggregate results
        if any(r == "CONTRADICTED" for r in verification_results):
            chain.is_verified = False
            chain.answer_text = (
                f"CONTRADICTED: Evidence directly refutes the claim.\n{chain.answer_text}"
            )
        elif (
            all(r == "VERIFIED" for r in verification_results) or "VERIFIED" in verification_results
        ):
            chain.is_verified = True
        else:
            chain.is_verified = False
            chain.answer_text = f"UNVERIFIABLE: Evidence is inconclusive.\n{chain.answer_text}"

        return chain
