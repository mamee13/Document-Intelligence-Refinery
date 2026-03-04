import hashlib
import json
import logging
import os
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
    Uses an LLM to extract structured facts from markdown tables.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.vlm_config = RULES.get("vlm", {})
        self.model = self.vlm_config.get("model_name", "google/gemini-2.0-flash-001")

    async def extract_facts_from_table(self, doc_id: str, table: Any) -> List[Dict[str, Any]]:
        """
        Parses a Markdown table into a list of structured facts.
        """
        if not self.api_key:
            return []

        prompt = f"""
        You are a financial data extractor. Convert the following markdown table into a JSON list of facts.
        Each fact must have: "fact_key", "fact_value", "unit", and "confidence".

        Document ID: {doc_id}
        Page: {table.page_number}
        Caption: {table.caption}

        Table:
        {table.markdown_grid}

        Return ONLY valid JSON array.
        """

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
                # Some models might return the root as a dict with a list field
                parsed = json.loads(raw_content)
                facts_list: List[Dict[str, Any]] = (
                    parsed.get("facts", parsed) if isinstance(parsed, dict) else parsed
                )

                # Add metadata
                for fact in facts_list:
                    fact["doc_id"] = doc_id
                    fact["page_number"] = table.page_number
                    fact["source_chunk_hash"] = hashlib.sha256(
                        table.markdown_grid.encode()
                    ).hexdigest()

                return facts_list
        except Exception as e:
            logger.error(f"Fact extraction failed: {str(e)}")
            return []
