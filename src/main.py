import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.agents.extractor import ExtractionRouter
from src.agents.refinery import SemanticRefinery
from src.agents.triage import TriageAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def process_corpus_refined(
    data_dir: Path, output_base: Path, file_filter: Optional[str] = None
) -> None:
    """
    Runs the Stage 2 pipeline: Triage -> Extraction -> Chunking -> Indexing -> Storage.
    """
    refinery = SemanticRefinery()

    # Get all PDFs in data/
    if file_filter:
        pdf_files = [data_dir / file_filter]
        if not pdf_files[0].exists():
            logger.error(f"File filter '{file_filter}' not found in {data_dir}")
            return
    else:
        pdf_files = list(data_dir.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir} for Deep Refining")

    for pdf_path in pdf_files[:12]:
        doc_id = pdf_path.stem
        try:
            await refinery.refine_document(pdf_path)
            logger.info(f"Deep Refining complete for {doc_id}")
        except Exception as e:
            logger.error(f"Failed to refine {doc_id}: {str(e)}")


def process_corpus_extract_only(
    data_dir: Path, output_base: Path, file_filter: Optional[str] = None
) -> None:
    """
    Runs the Stage 1 pipeline: Triage -> Extraction -> Ledger.
    """
    triage_agent = TriageAgent()
    router = ExtractionRouter()

    # Ensure directories exist
    profile_dir = output_base / "profiles"
    extracted_dir = output_base / "extracted"
    profile_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDFs in data/
    if file_filter:
        pdf_files = [data_dir / file_filter]
        if not pdf_files[0].exists():
            logger.error(f"File filter '{file_filter}' not found in {data_dir}")
            return
    else:
        pdf_files = list(data_dir.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir} for Extraction Only")

    for pdf_path in pdf_files[:12]:
        doc_id = pdf_path.stem
        logger.info(f"Processing {doc_id}...")

        profile = triage_agent.classify_document(pdf_path)
        extraction = router.route_and_extract(pdf_path, profile)

        extraction_path = extracted_dir / f"{doc_id}.json"
        with open(extraction_path, "w") as f:
            f.write(extraction.model_dump_json(indent=2))

        logger.info(f"Completed {doc_id} using {extraction.strategy_used}")


if __name__ == "__main__":
    import sys

    DATA_DIR = Path("data")
    OUTPUT_BASE = Path(".refinery")

    # Simple CLI dispatch
    mode = "extract"
    filter_val = None
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        filter_val = sys.argv[2]

    if mode == "refine":
        asyncio.run(process_corpus_refined(DATA_DIR, OUTPUT_BASE, file_filter=filter_val))
    else:
        process_corpus_extract_only(DATA_DIR, OUTPUT_BASE, file_filter=filter_val)
