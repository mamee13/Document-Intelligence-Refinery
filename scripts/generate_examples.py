import asyncio
import json
import logging
from pathlib import Path

from src.agents.query_agent import QueryAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(".refinery/examples")
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

QA_PAIRS = {
    "Audit Report - 2023": [
        "What is the primary purpose of this audit report?",
        "Who is the auditing authority for this document?",
        "What is the fiscal year covered by this audit?",
    ],
    "fta_performance_survey_final_report_2022": [
        "What is the main objective of the FTA initiative mentioned in the Executive Summary?",
        "List the names of the core team members who prepared the report.",
        "What regions or city administrations were visited as part of this survey?",
    ],
    "tax_expenditure_ethiopia_2021_22": [
        "What is the total tax expenditure estimate for the year 2021/22?",
        "What are the main types of taxes covered in this report?",
        "Identify a specific sector that benefited from tax exemptions according to the first 10 pages.",
    ],
    "CBE ANNUAL REPORT 2023-24": [
        "What is the theme of the 2023-24 Annual Report?",
        "Who is the Chairperson of the Board of Directors?",
        "Mention one key financial highlight from the President's message.",
    ],
}


async def generate_examples() -> None:
    agent = QueryAgent()
    count = 0

    for doc_id, questions in QA_PAIRS.items():
        logger.info(f"Generating examples for {doc_id}...")
        for i, query in enumerate(questions):
            try:
                # We append the document name to the query to guide the agent
                full_query = f"In the document '{doc_id}', {query}"
                result = await agent.run(full_query)

                output_file = EXAMPLES_DIR / f"{doc_id}_q{i+1}.json"
                with open(output_file, "w") as f:
                    f.write(result.model_dump_json(indent=2))

                logger.info(f"Saved: {output_file.name}")
                count += 1
            except Exception as e:
                logger.error(f"Failed to generate example {i+1} for {doc_id}: {str(e)}")

    logger.info(f"Successfully generated {count} example Q&A files.")


if __name__ == "__main__":
    asyncio.run(generate_examples())
