import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexManager
from src.agents.triage import TriageAgent
from src.data.fact_table import FactTable
from src.data.vector_store import VectorStore
from src.models.core import LDU, ExtractedDocument, PageIndex

logger = logging.getLogger(__name__)


class SemanticRefinery:
    """
    Orchestrates the full multi-tier semantic refining process.
    PDF -> Profile -> Extraction -> Chunks (LDUs) -> Index -> Vector Store + Fact Table.
    """

    def __init__(self) -> None:
        self.triage = TriageAgent()
        self.router = ExtractionRouter()
        self.chunker = ChunkingEngine()
        self.indexer = PageIndexManager()
        self.vector_store = VectorStore()
        self.fact_table = FactTable()

    async def refine_document(self, pdf_path: Path, max_pages: Optional[int] = None) -> PageIndex:
        """
        Runs a single document through the full semantic refinery.
        """
        doc_id = pdf_path.stem
        logger.info(f"Refining document: {doc_id}")

        # 1. Triage
        profile = self.triage.classify_document(pdf_path)
        logger.info(f"Triage complete: {profile.origin_type}, {profile.layout_complexity}")

        # 2. Extract
        extraction: ExtractedDocument = self.router.route_and_extract(
            pdf_path, profile, max_pages=max_pages
        )
        logger.info(f"Extraction complete using {extraction.strategy_used}")

        # 3. Chunk
        ldus: List[LDU] = self.chunker.chunk(extraction)
        logger.info(f"Semantic chunking complete: {len(ldus)} LDUs generated")

        # 4. Index
        index = await self.indexer.create_index(doc_id, ldus)
        logger.info(f"PageIndex generated with {len(index.root_nodes)} sections")

        # 5. Store - Vector Store
        self.vector_store.ingest_ldus(ldus)
        logger.info("LDUs ingested into Vector Store")

        # 6. Store - Fact Table (if applicable)
        # For now, we manually extract facts from tables in a simplified way
        for table in extraction.tables:
            # In a more advanced version, we'd use an LLM or pattern matcher
            # to extract keys/values from the markdown_grid
            fact_data = {
                "doc_id": doc_id,
                "page_number": table.page_number,
                "fact_key": "Extracted Table",
                "fact_value": table.markdown_grid[:100] + "...",  # Placeholder
                "unit": "Markdown",
                "confidence": extraction.confidence_score,
                "source_chunk_hash": hashlib.sha256(table.markdown_grid.encode()).hexdigest(),
            }
            self.fact_table.insert_fact(fact_data)

        if extraction.tables:
            logger.info(f"Stored {len(extraction.tables)} tables in Fact Table")

        return index
