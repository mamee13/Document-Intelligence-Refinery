import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class BaseDomainClassifier(ABC):
    """
    Interface for domain classification strategies.
    """

    @abstractmethod
    def classify(self, text: str) -> Optional[str]:
        """
        Analyzes text and returns a domain hint if detected.
        """
        pass


class KeywordDomainClassifier(BaseDomainClassifier):
    """
    Simple keyword-based domain classification.
    """

    def __init__(self, domain_hints: Dict[str, List[str]]):
        self.domain_hints = domain_hints

    def classify(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for domain, keywords in self.domain_hints.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    return str(domain)
        return None


class VLMDomainClassifier(BaseDomainClassifier):
    """
    VLM-based domain classification using a LLM.
    """

    def __init__(self, model_name: str = "google/gemini-2.0-flash-001"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0,
        )

    def classify(self, text: str) -> Optional[str]:
        if not text.strip():
            return None

        system_prompt = (
            "You are a document classifier. Given a text snippet, "
            "determine if it belongs to one of these domains: "
            "financial, tax, annual_report. "
            "Return ONLY the domain name or 'unknown'."
        )

        try:
            # We only send first 2000 chars for efficiency
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=text[:2000])]
            )
            domain = response.content.strip().lower()
            return domain if domain != "unknown" else None
        except Exception:
            return None
