import pytest
import shutil
from pathlib import Path

from src.agents.triage import TriageAgent
from src.models.core import DocumentProfile

# Assuming tests are run from project root where data/ exists
DATA_DIR = Path("data")

# Known Ground Truth Documents from the corpus
DOCS = {
    "A": DATA_DIR / "CBE ANNUAL REPORT 2023-24.pdf",
    "B": DATA_DIR / "Audit Report - 2023.pdf",
    "C": DATA_DIR / "fta_performance_survey_final_report_2022.pdf",
    "D": DATA_DIR / "tax_expenditure_ethiopia_2021_22.pdf",
}


@pytest.fixture(scope="module", autouse=True)
def cleanup_profiles():
    """Ensure a clean profiles directory before/after tests."""
    prof_dir = Path(".refinery/profiles")
    if prof_dir.exists():
        shutil.rmtree(prof_dir)
    prof_dir.mkdir(parents=True, exist_ok=True)
    yield
    # We leave the output files to be manually inspected


def test_triage_class_a():
    """
    Class A (Native Digital) should be native_digital
    + multi_col or table_heavy
    """
    if not DOCS["A"].exists():
        pytest.skip(f"Corpus file not found: {DOCS['A']}")

    agent = TriageAgent(max_pages_to_analyze=10)
    profile = agent.classify_document(DOCS["A"])

    assert profile.doc_id == DOCS["A"].stem
    assert profile.origin_type == "native_digital"
    # Class A has heavy multi-col formatting
    assert profile.layout_complexity in ["multi_column", "table_heavy"]
    assert profile.domain_hint in ["financial", "annual_report"]


def test_triage_class_b():
    """Class B (Scanned) should be flagged as scanned_image"""
    if not DOCS["B"].exists():
        pytest.skip(f"Corpus file not found: {DOCS['B']}")

    agent = TriageAgent(max_pages_to_analyze=10)
    profile = agent.classify_document(DOCS["B"])

    assert profile.origin_type == "scanned_image"
    # Scanned documents default to single_column before layout analysis
    assert profile.layout_complexity == "single_column"


def test_triage_class_c():
    """Class C (Mixed/Complex) has tables and columns"""
    if not DOCS["C"].exists():
        pytest.skip(f"Corpus file not found: {DOCS['C']}")

    agent = TriageAgent(max_pages_to_analyze=10)
    profile = agent.classify_document(DOCS["C"])

    assert profile.origin_type == "native_digital"
    # Class C has 4 naive tables detected
    assert profile.layout_complexity in ["table_heavy", "multi_column"]


def test_triage_class_d():
    """Class D (Structured) has columns"""
    if not DOCS["D"].exists():
        pytest.skip(f"Corpus file not found: {DOCS['D']}")

    agent = TriageAgent(max_pages_to_analyze=10)
    profile = agent.classify_document(DOCS["D"])

    assert profile.origin_type == "native_digital"
    assert profile.layout_complexity == "multi_column"
    assert profile.domain_hint in ["tax", "financial"]


def test_profile_artifact_written():
    """Check that the physical file is written to .refinery/profiles"""
    if not DOCS["B"].exists():
        pytest.skip("Test requires corpus file")

    agent = TriageAgent(max_pages_to_analyze=2)
    _ = agent.classify_document(DOCS["B"])

    expected_path = Path(".refinery/profiles") / f"{DOCS['B'].stem}.json"
    assert expected_path.exists()

    # Verify it parses back correctly
    parsed = DocumentProfile.model_validate_json(expected_path.read_text())
    assert parsed.origin_type == "scanned_image"
