import logging
from pathlib import Path
from typing import Optional

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_corpus(data_dir: Path, output_base: Path, file_filter: Optional[str] = None) -> None:
    """
    Runs the full pipeline: Triage -> Extraction -> Ledger.
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

    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")

    # Limit to 12 for Day 2 verification if needed, or just process all
    # The plan specifies 12 documents.
    for pdf_path in pdf_files[:12]:
        doc_id = pdf_path.stem
        logger.info(f"Processing {doc_id}...")

        # 1. Triage (classify_document also saves the profile to .refinery/profiles/)
        profile = triage_agent.classify_document(pdf_path)

        # 2. Extract
        # For batch processing, we use full extraction (no max_pages limit)
        # Note: VisionExtractor will use budget guard
        extraction = router.route_and_extract(pdf_path, profile)

        extraction_path = extracted_dir / f"{doc_id}.json"
        with open(extraction_path, "w") as f:
            f.write(extraction.model_dump_json(indent=2))

        logger.info(f"Completed {doc_id} using {extraction.strategy_used}")


if __name__ == "__main__":
    DATA_DIR = Path("data")
    OUTPUT_BASE = Path(".refinery")
    process_corpus(DATA_DIR, OUTPUT_BASE)
