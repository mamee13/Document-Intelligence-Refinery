import hashlib
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import RULES

logger = logging.getLogger(__name__)


class FactTable:
    """
    Manages the SQLite database for structured facts extracted from tables.
    """

    def __init__(self, db_path: str = ".refinery/refinery_facts.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Core facts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page_number INTEGER,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                unit TEXT,
                confidence REAL,
                source_chunk_hash TEXT,
                metadata TEXT,  -- JSON string for extra info
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Indexing for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON facts(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_key ON facts(fact_key)")

        conn.commit()
        conn.close()

    def insert_fact(self, fact_data: Dict[str, Any]) -> int:
        """Inserts a single fact and returns its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        keys = [
            "doc_id",
            "page_number",
            "fact_key",
            "fact_value",
            "unit",
            "confidence",
            "source_chunk_hash",
            "metadata",
        ]
        values = [fact_data.get(k) for k in keys]

        # Convert metadata to string if dict
        if isinstance(values[-1], dict):
            import json

            values[-1] = json.dumps(values[-1])

        cursor.execute(
            f"""
            INSERT INTO facts ({", ".join(keys)})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            values,
        )

        row_id = cursor.lastrowid or -1
        conn.commit()
        conn.close()
        return row_id

    def query_facts(
        self, doc_id: Optional[str] = None, fact_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Queries facts based on filters."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM facts WHERE 1=1"
        params = []
        if doc_id:
            query += " AND doc_id = ?"
            params.append(doc_id)
        if fact_key:
            query += " AND fact_key LIKE ?"
            params.append(f"%{fact_key}%")

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows


class FactTableExtractor:
    """
    Uses an LLM to extract structured facts from markdown tables with robust fallbacks.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.vlm_config = RULES.get("vlm", {})
        self.model = self.vlm_config.get("model_name", "google/gemini-2.0-flash-001")
        self.max_retries = 2

    def _is_valid_fact(self, fact: Dict[str, Any]) -> bool:
        """
        Validates if a fact is actually a quantitative/numerical fact.
        Filters out table of contents, headers, and non-factual data.
        """
        key = str(fact.get("fact_key", "")).lower()
        value = str(fact.get("fact_value", "")).lower()

        # Filter out common non-facts
        invalid_patterns = [
            r"^(page|section|chapter|table of contents|contents|index)",
            r"^(i{1,5}|[ivxlcdm]+)$",  # Roman numerals alone
            r"^(profile|vision|mission|motto|values|note)$",
            r"^\d+$",  # Just page numbers
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, key) or re.match(pattern, value):
                return False

        # Must have some numeric content or be a meaningful key-value pair
        has_number = bool(re.search(r"\d", value))
        has_meaningful_key = len(key) > 3 and not key.isdigit()

        return has_number and has_meaningful_key

    def _parse_table_directly(self, table: Any, doc_id: str) -> List[Dict[str, Any]]:
        """
        Fallback: Direct parsing of markdown table without LLM.
        Extracts key-value pairs from tables with numerical data.
        """
        facts: List[Dict[str, Any]] = []
        markdown = getattr(table, "markdown_grid", "")

        if not markdown or "|" not in markdown:
            return facts

        lines = [line.strip() for line in markdown.split("\n") if line.strip()]

        # Separate header/separator from data
        data_start_idx = 0
        for i, line in enumerate(lines):
            # Find the separator line (contains dashes)
            if re.match(r"^\|[\s\-:]+\|", line):
                data_start_idx = i + 1
                break

        # Process data rows
        for line in lines[data_start_idx:]:
            if not line.startswith("|"):
                continue

            cells = [cell.strip() for cell in line.split("|")]
            # Remove empty cells from split
            cells = [c for c in cells if c]

            if len(cells) < 2:
                continue

            # Extract key-value pairs
            key = cells[0]
            value = " | ".join(cells[1:]) if len(cells) > 2 else cells[1]

            # Skip if value doesn't contain numbers
            if not re.search(r"\d", value):
                continue

            # Skip common header words
            if key.lower() in ["item", "description", "key", "name", "field", "metric"]:
                continue

            fact = {
                "fact_key": key,
                "fact_value": value,
                "unit": self._extract_unit(value),
                "confidence": 0.75,  # Medium confidence for direct parsing
                "doc_id": doc_id,
                "page_number": getattr(table, "page_number", 0),
                "source_chunk_hash": hashlib.sha256(markdown.encode()).hexdigest(),
            }

            if self._is_valid_fact(fact):
                # Add bbox metadata if available
                bbox = getattr(table, "bbox", None)
                if bbox:
                    fact["metadata"] = json.dumps(
                        {"bbox": (bbox.model_dump() if hasattr(bbox, "model_dump") else str(bbox))}
                    )
                facts.append(fact)

        return facts

    def _extract_unit(self, value: str) -> Optional[str]:
        """Extracts unit from a value string."""
        # Common units
        units = [
            "USD",
            "EUR",
            "GBP",
            "Birr",
            "%",
            "kg",
            "g",
            "ton",
            "m",
            "km",
            "million",
            "billion",
            "thousand",
        ]

        value_lower = value.lower()
        for unit in units:
            if unit.lower() in value_lower:
                return unit

        return None

    async def extract_facts_from_table(self, doc_id: str, table: Any) -> List[Dict[str, Any]]:
        """
        Parses a Markdown table into structured facts with robust error handling.
        """
        if not self.api_key:
            logger.warning("No API key, using direct parsing fallback")
            return self._parse_table_directly(table, doc_id)

        markdown = getattr(table, "markdown_grid", "")

        # Skip empty or very small tables
        if not markdown or len(markdown) < 20:
            return []

        # Skip tables that look like table of contents
        if any(
            keyword in markdown.lower() for keyword in ["contents", "page", "section", "chapter"]
        ):
            logger.debug(f"Skipping TOC-like table in {doc_id}")
            return []

        prompt = f"""Extract ONLY quantitative facts from this table. Focus on:
- Financial figures (revenue, profit, assets, etc.)
- Statistical data (percentages, counts, measurements)
- Key performance indicators

IGNORE:
- Table of contents entries
- Page numbers
- Section headers
- Non-numerical metadata

Return JSON with "facts" array. Each fact needs:
- fact_key: Descriptive name (e.g. "Total Revenue 2023")
- fact_value: The numeric value with context
- unit: Unit of measure (USD, %, kg, etc.) or null
- confidence: 0.0-1.0 score

Document: {doc_id}
Page: {getattr(table, "page_number", 0)}

Table:
{markdown[:2000]}

Return ONLY valid JSON."""

        # Try LLM extraction with retries
        for attempt in range(self.max_retries):
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"},
                        },
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    raw_content = data["choices"][0]["message"]["content"]

                    # Robust JSON parsing
                    parsed = self._parse_json_response(raw_content)

                    facts_list: List[Dict[str, Any]] = []
                    candidate = parsed.get("facts", parsed) if isinstance(parsed, dict) else parsed

                    if isinstance(candidate, list):
                        for item in candidate:
                            if not isinstance(item, dict):
                                continue

                            key = item.get("fact_key")
                            val = item.get("fact_value")

                            if key and val:
                                # Add provenance metadata
                                item["doc_id"] = doc_id
                                item["page_number"] = getattr(table, "page_number", 0)
                                item["source_chunk_hash"] = hashlib.sha256(
                                    markdown.encode()
                                ).hexdigest()

                                # Store BBox in metadata
                                bbox = getattr(table, "bbox", None)
                                if bbox:
                                    item["metadata"] = json.dumps(
                                        {
                                            "bbox": (
                                                bbox.model_dump()
                                                if hasattr(bbox, "model_dump")
                                                else str(bbox)
                                            )
                                        }
                                    )

                                # Validate fact quality
                                if self._is_valid_fact(item):
                                    facts_list.append(item)
                                else:
                                    logger.debug(f"Filtered out low-quality fact: {key}={val}")

                    if facts_list:
                        logger.info(
                            f"Extracted {len(facts_list)} valid facts from table in {doc_id}"
                        )
                        return facts_list

                    # If LLM returned empty, try direct parsing
                    logger.warning(f"LLM returned no valid facts, trying direct parsing")
                    return self._parse_table_directly(table, doc_id)

            except Exception as e:
                logger.warning(
                    f"Fact extraction attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                )
                if attempt == self.max_retries - 1:
                    # Final fallback
                    logger.info("Using direct parsing fallback")
                    return self._parse_table_directly(table, doc_id)

        return []

    def _parse_json_response(self, raw_content: str) -> Any:
        """Robustly parses JSON from LLM response."""
        try:
            # Strip markdown code blocks
            fc = raw_content.strip()
            if fc.startswith("```"):
                fc = re.sub(r"```(json)?", "", fc).strip()

            return json.loads(fc)
        except json.JSONDecodeError:
            # Try fixing common escaping issues
            pattern = r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})'
            fixed_content = re.sub(pattern, r"\\\\", raw_content.strip())
            return json.loads(fixed_content)
