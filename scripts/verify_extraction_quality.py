#!/usr/bin/env python3
"""
Verification script for extraction quality analysis.
Provides ground truth validation for the final report.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def verify_fact_precision() -> Dict[str, Any]:
    """Manually verify a sample of extracted facts for precision."""
    conn = sqlite3.connect(".refinery/refinery_facts.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get random sample of 100 facts with units
    cursor.execute(
        """
        SELECT fact_key, fact_value, unit, doc_id, page_number
        FROM facts
        WHERE unit IS NOT NULL AND unit != 'page_number'
        ORDER BY RANDOM()
        LIMIT 100
    """
    )

    sample = [dict(row) for row in cursor.fetchall()]

    # Manual validation criteria
    valid_count = 0
    invalid_count = 0

    for fact in sample:
        key = fact["fact_key"].lower()
        value = fact["fact_value"].lower()

        # Check if it's a real quantitative fact
        is_valid = True

        # Invalid patterns
        invalid_keywords = ["page", "section", "chapter", "contents"]
        if any(word in key for word in invalid_keywords):
            is_valid = False
        if any(word in value for word in ["page", "section", "chapter"]):
            is_valid = False
        if key.isdigit() or value.isdigit():
            is_valid = False

        # Must have meaningful content
        if len(key) < 3 or len(value) < 1:
            is_valid = False

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    precision = valid_count / len(sample) if sample else 0

    conn.close()

    return {
        "sample_size": len(sample),
        "valid_facts": valid_count,
        "invalid_facts": invalid_count,
        "precision": precision,
        "precision_percent": f"{precision * 100:.1f}%",
    }


def verify_table_extraction() -> Dict[str, Any]:
    """Verify table extraction across document classes."""
    results = {}

    test_docs = {
        "Class B (Scanned)": ["2021_Audited_Financial_Statement_Report"],
        "Class C (Multi-column)": ["reading_notes"],
        "Class D (Table-heavy)": [
            "tax_expenditure_ethiopia_2021_22",
            "Annual_Report_JUNE-2023",
        ],
    }

    for doc_class, docs in test_docs.items():
        class_results = []
        for doc_id in docs:
            try:
                with open(f".refinery/extracted/{doc_id}.json") as f:
                    data = json.load(f)
                    class_results.append(
                        {
                            "doc_id": doc_id,
                            "tables": len(data["tables"]),
                            "text_blocks": len(data["text_blocks"]),
                            "confidence": data["confidence_score"],
                        }
                    )
            except FileNotFoundError:
                pass

        results[doc_class] = class_results

    return results


def verify_pageindex_coverage() -> Dict[str, Any]:
    """Verify PageIndex content coverage."""
    pageindex_files = list(Path(".refinery/pageindex").glob("*.json"))

    total_sections = 0
    sections_with_content = 0
    sections_with_summaries = 0

    for filepath in pageindex_files:
        with open(filepath) as f:
            data = json.load(f)

            def count_sections(nodes: List[Dict[str, Any]]) -> None:
                nonlocal total_sections, sections_with_content, sections_with_summaries
                for node in nodes:
                    total_sections += 1
                    if node.get("data_types_present"):
                        sections_with_content += 1
                    if (
                        node.get("summary")
                        and node["summary"] != "No content found for this section."
                    ):
                        sections_with_summaries += 1
                    if node.get("children"):
                        count_sections(node["children"])

            count_sections(data["root_nodes"])

    return {
        "total_sections": total_sections,
        "sections_with_content": sections_with_content,
        "sections_with_summaries": sections_with_summaries,
        "content_coverage": (
            f"{(sections_with_content / total_sections * 100):.1f}%" if total_sections else "0%"
        ),
        "summary_coverage": (
            f"{(sections_with_summaries / total_sections * 100):.1f}%" if total_sections else "0%"
        ),
    }


def main() -> None:
    print("=" * 60)
    print("EXTRACTION QUALITY VERIFICATION REPORT")
    print("=" * 60)

    # Fact precision
    print("\n1. FACT EXTRACTION PRECISION")
    print("-" * 60)
    fact_results = verify_fact_precision()
    print(f"Sample Size: {fact_results['sample_size']} facts")
    print(f"Valid Facts: {fact_results['valid_facts']}")
    print(f"Invalid Facts: {fact_results['invalid_facts']}")
    print(f"Precision: {fact_results['precision_percent']}")

    # Table extraction
    print("\n2. TABLE EXTRACTION BY DOCUMENT CLASS")
    print("-" * 60)
    table_results = verify_table_extraction()
    for doc_class, docs in table_results.items():
        print(f"\n{doc_class}:")
        for doc in docs:
            print(f"  {doc['doc_id'][:50]}:")
            print(
                f"    Tables: {doc['tables']}, "
                f"Blocks: {doc['text_blocks']}, "
                f"Confidence: {doc['confidence']}"
            )

    # PageIndex coverage
    print("\n3. PAGEINDEX CONTENT COVERAGE")
    print("-" * 60)
    coverage_results = verify_pageindex_coverage()
    print(f"Total Sections: {coverage_results['total_sections']}")
    print(f"Sections with Content: {coverage_results['sections_with_content']}")
    print(f"Sections with Summaries: {coverage_results['sections_with_summaries']}")
    print(f"Content Coverage: {coverage_results['content_coverage']}")
    print(f"Summary Coverage: {coverage_results['summary_coverage']}")

    # Database stats
    print("\n4. DATABASE STATISTICS")
    print("-" * 60)
    conn = sqlite3.connect(".refinery/refinery_facts.db")
    cursor = conn.cursor()

    total_facts = cursor.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    quantitative_facts = cursor.execute(
        """
        SELECT COUNT(*) FROM facts
        WHERE unit IS NOT NULL AND unit != 'page_number'
        """
    ).fetchone()[0]

    print(f"Total Facts: {total_facts}")
    print(f"Quantitative Facts: {quantitative_facts}")
    print(f"Percentage Quantitative: " f"{(quantitative_facts / total_facts * 100):.1f}%")

    conn.close()

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
