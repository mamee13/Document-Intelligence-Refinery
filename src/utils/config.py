from pathlib import Path
from typing import Any

import yaml


def load_extraction_rules() -> dict[str, Any]:
    """Load the extraction rules and thresholds from the yaml file."""
    # Assuming this is run from the project root
    rules_path = Path("rubric/extraction_rules.yaml")
    if not rules_path.exists():
        # Fallback if run from a different directory
        parent = Path(__file__).parent.parent.parent
        rules_path = parent / "rubric" / "extraction_rules.yaml"

    with open(rules_path, "r") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


# Singleton load so it's ready for any model/agent
RULES = load_extraction_rules()
