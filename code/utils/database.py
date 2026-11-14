from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS inferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    patient_name TEXT,
    patient_id TEXT,
    patient_age TEXT,
    patient_sex TEXT,
    input_image_path TEXT NOT NULL,
    prediction_json TEXT NOT NULL,
    pdf_path TEXT
);
"""


class InferenceDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(DB_SCHEMA)
            conn.commit()

    def insert_inference(
        self,
        created_at: str,
        patient: Dict[str, str],
        input_image_path: str,
        prediction: Dict[str, Any],
        pdf_path: Optional[str],
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO inferences (
                    created_at, patient_name, patient_id, patient_age, patient_sex,
                    input_image_path, prediction_json, pdf_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    patient.get("name", ""),
                    patient.get("id", ""),
                    patient.get("age", ""),
                    patient.get("sex", ""),
                    input_image_path,
                    json.dumps(prediction),
                    pdf_path,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_pdf_path(self, inference_id: int, pdf_path: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE inferences SET pdf_path = ? WHERE id = ?", (pdf_path, inference_id))
            conn.commit()

    def get_inference(self, inference_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM inferences WHERE id = ?", (inference_id,)).fetchone()
            if row is None:
                return None
            payload = dict(row)
            payload["prediction_json"] = json.loads(payload["prediction_json"])
            return payload
