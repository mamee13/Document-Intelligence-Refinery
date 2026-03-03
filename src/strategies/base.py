from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.models.core import ExtractedDocument


class BaseExtractor(ABC):
    """
    Base class for all extraction strategies.
    Ensures a unified interface and output schema.
    """

    @abstractmethod
    def extract(
        self, pdf_path: Path, doc_id: str, max_pages: Optional[int] = None
    ) -> ExtractedDocument:
        """
        Extracts content from a PDF and returns an ExtractedDocument object.
        """
        pass
