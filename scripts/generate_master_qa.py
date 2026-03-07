import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from src.agents.audit import AuditManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3 Questions per doc class
QA_TASKS = [
    # Audit Reports
    {
        "doc": "Audit Report - 2023",
        "q": "What was the total audit coverage for 2023?",
    },
    {
        "doc": "Audit Report - 2023",
        "q": "Identify any major findings related to procurement.",
    },
    {
        "doc": "Audit Report - 2023",
        "q": "Summarize the recommendation for the finance department.",
    },
    # 2022 Audited Financial Statement Report (Class B)
    {
        "doc": "2022_Audited_Financial_Statement_Report",
        "q": "Who were the independent auditors for the year 2022?",
    },
    {
        "doc": "2022_Audited_Financial_Statement_Report",
        "q": "What was the total comprehensive income for the year ended 31 December 2022?",
    },
    {
        "doc": "2022_Audited_Financial_Statement_Report",
        "q": "Did the auditors express an 'unqualified' or 'qualified' opinion?",
    },
    # 2021 Audited Financial Statement Report (Class B)
    {
        "doc": "2021_Audited_Financial_Statement_Report",
        "q": "What was the total revenue for the year ended 30 June 2021?",
    },
    {
        "doc": "2021_Audited_Financial_Statement_Report",
        "q": "Identify the total operating expenses for the year ended 30 June 2021.",
    },
    {
        "doc": "2021_Audited_Financial_Statement_Report",
        "q": "What was the profit after tax for FY 2020/21?",
    },
    # CBE Annual Reports
    {
        "doc": "CBE ANNUAL REPORT 2023-24",
        "q": "What was CBE's total asset value in 2023-24?",
    },
    {
        "doc": "CBE ANNUAL REPORT 2023-24",
        "q": "How much was the net profit before tax?",
    },
    {
        "doc": "CBE ANNUAL REPORT 2023-24",
        "q": "What are the strategic initiatives mentioned?",
    },
    # CBE Annual Report June 2023 (Class A)
    {
        "doc": "Annual_Report_JUNE-2023",
        "q": "What was the total deposit position by the end of June 2023?",
    },
    {
        "doc": "Annual_Report_JUNE-2023",
        "q": "Identify the number of branches as of June 30, 2023.",
    },
    {
        "doc": "Annual_Report_JUNE-2023",
        "q": "What was the total revenue recorded for the 2022/23 fiscal year?",
    },
    # CBE Annual Report June 2022 (Class A)
    {
        "doc": "Annual_Report_JUNE-2022",
        "q": "What was the profit before tax for the year ended June 2022?",
    },
    {
        "doc": "Annual_Report_JUNE-2022",
        "q": "How many active domestic customers did CBE have at the end of June 2022?",
    },
    {
        "doc": "Annual_Report_JUNE-2022",
        "q": "What was the total asset value reported for the year ended June 30, 2022?",
    },
    # Consumer Price Index (Class D)
    {
        "doc": "Consumer Price Index July 2025",
        "q": "What was the year-on-year general inflation rate for July EFY 2017?",
    },
    {
        "doc": "Consumer Price Index July 2025",
        "q": "How much did the Food and Non-alcoholic Beverages index increase in July 2017 compared to the previous year?",
    },
    {
        "doc": "Consumer Price Index July 2025",
        "q": "Briefly explain the trend of annual inflation in Ethiopia as presented in the report.",
    },
    {
        "doc": "Consumer Price Index March 2025",
        "q": "What was the general inflation rate for March EFY 2017?",
    },
    {
        "doc": "Consumer Price Index March 2025",
        "q": "Compare the food inflation rate between March 2017 and March 2016.",
    },
    {
        "doc": "Consumer Price Index March 2025",
        "q": "Which major food items contributed to the inflation in March 2017?",
    },
    # Pharmaceutical Manufacturing Opportunities (Class C)
    {
        "doc": "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF",
        "q": "What is the estimated market size for pharmaceuticals in Ethiopia according to the report?",
    },
    {
        "doc": "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF",
        "q": "List three main investment incentives provided by the Ethiopian government for pharmaceutical manufacturers.",
    },
    {
        "doc": "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF",
        "q": "What percentage of the Ethiopian pharmaceutical market is currently met by local production?",
    },
    # Security Vulnerability Disclosure Procedure (Class C)
    {
        "doc": "Security_Vulnerability_Disclosure_Standard_Procedure_1",
        "q": "What is the defined scope of the Security Vulnerability Disclosure Policy?",
    },
    {
        "doc": "Security_Vulnerability_Disclosure_Standard_Procedure_1",
        "q": "What are the specific steps a researcher must follow to report a vulnerability?",
    },
    {
        "doc": "Security_Vulnerability_Disclosure_Standard_Procedure_1",
        "q": "What are the 'Guidelines for Responsible Disclosure' mentioned in the document?",
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


async def generate_examples(filter_doc: Optional[str] = None) -> None:
    manager = AuditManager()
    output_dir = Path(".refinery/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_tasks = [t for t in QA_TASKS if not filter_doc or filter_doc in t["doc"]]

    for i, task in enumerate(filtered_tasks):
        logger.info(f"Generating Master QA: {task['q']} for {task['doc']}")
        try:
            result = await manager.verify_claim(task["q"], doc_id=task["doc"])
            # Use original index logic for naming if possible, or just question hash
            doc_tasks = [t for t in QA_TASKS if t["doc"] == task["doc"]]
            q_idx = doc_tasks.index(task) + 1
            filename = f"{task['doc']}_master_q{q_idx}.json"
            with open(output_dir / filename, "w") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info(f"Saved {filename}")
        except Exception as e:
            logger.error(f"Failed task: {e}")


if __name__ == "__main__":
    doc_filter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--file" else None
    asyncio.run(generate_examples(doc_filter))
