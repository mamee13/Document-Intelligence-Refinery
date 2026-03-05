import asyncio
import logging
from pathlib import Path

from src.agents.audit import AuditManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3 Questions per doc class
QA_TASKS = [
    # Audit Reports
    {"doc": "Audit Report - 2023", "q": "What was the total audit coverage for 2023?"},
    {"doc": "Audit Report - 2023", "q": "Identify any major findings related to procurement."},
    {"doc": "Audit Report - 2023", "q": "Summarize the recommendation for the finance department."},
    # CBE Annual Reports
    {"doc": "CBE ANNUAL REPORT 2023-24", "q": "What was CBE's total asset value in 2023-24?"},
    {"doc": "CBE ANNUAL REPORT 2023-24", "q": "How much was the net profit before tax?"},
    {"doc": "CBE ANNUAL REPORT 2023-24", "q": "What are the strategic initiatives mentioned?"},
    # Performance Surveys
    {
        "doc": "fta_performance_survey_final_report_2022",
        "q": "What is the overall satisfaction score?",
    },
    {
        "doc": "fta_performance_survey_final_report_2022",
        "q": "Which department had the lowest performance?",
    },
    {
        "doc": "fta_performance_survey_final_report_2022",
        "q": "What were the key recommendations from the survey?",
    },
    # Tax Expenditures
    {
        "doc": "tax_expenditure_ethiopia_2021_22",
        "q": "What was the total tax expenditure in 2021-22?",
    },
    {
        "doc": "tax_expenditure_ethiopia_2021_22",
        "q": "Which sector received the highest tax incentive?",
    },
    {
        "doc": "tax_expenditure_ethiopia_2021_22",
        "q": "How does tax expenditure compare to previous years?",
    },
]


async def generate_examples() -> None:
    manager = AuditManager()
    output_dir = Path(".refinery/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, task in enumerate(QA_TASKS):
        logger.info(f"Generating Master QA {i+1}/12: {task['q']} for {task['doc']}")
        try:
            result = await manager.verify_claim(task["q"], doc_id=task["doc"])
            filename = f"{task['doc']}_master_q{i%3 + 1}.json"
            with open(output_dir / filename, "w") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info(f"Saved {filename}")
        except Exception as e:
            logger.error(f"Failed task {i+1}: {e}")


if __name__ == "__main__":
    asyncio.run(generate_examples())
