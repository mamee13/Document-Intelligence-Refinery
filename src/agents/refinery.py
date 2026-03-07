import logging
from pathlib import Path
from typing import List, Optional

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexManager
from src.agents.triage import TriageAgent
from src.data.fact_table import FactTable, FactTableExtractor
from src.data.vector_store import VectorStore
from src.models.core import LDU, ExtractedDocument, PageIndex

logger = logging.getLogger(__name__)


class SemanticRefinery:
    """
    Orchestrates the full multi-tier semantic refining process.
    PDF -> Profile -> Extraction -> Chunks -> Index -> Store.
    """

    def __init__(self) -> None:
        self.triage = TriageAgent()
        self.router = ExtractionRouter()
        self.chunker = ChunkingEngine()
        self.indexer = PageIndexManager()
        self.vector_store = VectorStore()
        self.fact_table = FactTable()
        self.fact_extractor = FactTableExtractor()

    async def refine_document(self, pdf_path: Path, max_pages: Optional[int] = None) -> PageIndex:
        """
        Runs a single document through the full semantic refinery.
        """
        doc_id = pdf_path.stem
        logger.info(f"Refining document: {doc_id}")

        # 1. Triage
        profile = self.triage.classify_document(pdf_path)
        logger.info(f"Triage complete: {profile.origin_type}, " f"{profile.layout_complexity}")

        # 2. Extract
        extraction: ExtractedDocument = self.router.route_and_extract(
            pdf_path, profile, max_pages=max_pages
        )
        logger.info(f"Extraction complete using {extraction.strategy_used}")

        # Persist raw extraction for submission compliance
        extracted_dir = Path(".refinery/extracted")
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with open(extracted_dir / f"{doc_id}.json", "w") as f:
            f.write(extraction.model_dump_json(indent=2))
        logger.info(f"Raw extraction persisted to {extracted_dir}/{doc_id}.json")

        # 3. Chunk
        ldus: List[LDU] = self.chunker.chunk(extraction)
        logger.info(f"Semantic chunking complete: {len(ldus)} LDUs generated")

        # 4. Index
        index = await self.indexer.create_index(doc_id, ldus)
        logger.info(f"PageIndex generated with {len(index.root_nodes)} sections")

        # 5. Store - Vector Store
        self.vector_store.ingest_ldus(ldus)
        logger.info("LDUs ingested into Vector Store")

        # 6. Store - Fact Table
        for table in extraction.tables:
            facts = await self.fact_extractor.extract_facts_from_table(doc_id, table)
            for fact in facts:
                self.fact_table.insert_fact(fact)

        if extraction.tables:
            logger.info(
                f"Structured fact extraction attempted for " f"{len(extraction.tables)} tables"
            )

        return index
