from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.database import InferenceDB
from code.utils.reporting import generate_pdf_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/re-generate PDF report from a stored inference record.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--inference-id", type=int, required=True)
    parser.add_argument("--db-path", type=str, default="app/dr_inference.db")
    parser.add_argument("--out", type=str, default="results/reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    db = InferenceDB(project_root / args.db_path)
    row = db.get_inference(args.inference_id)
    if row is None:
        raise RuntimeError(f"Inference id {args.inference_id} not found in DB")

    pred = row["prediction_json"]
    artifacts = pred.get("artifacts", {})
    patient = {
        "name": row.get("patient_name", ""),
        "id": row.get("patient_id", ""),
        "age": row.get("patient_age", ""),
        "sex": row.get("patient_sex", ""),
    }

    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"report_{args.inference_id}.pdf"

    pdf_path = generate_pdf_report(
        output_path=output_path,
        patient=patient,
        inference=pred,
        artifact_paths=artifacts,
        model_summary=pred.get("model_version", {}),
    )

    db.update_pdf_path(args.inference_id, pdf_path)
    print(json.dumps({"inference_id": args.inference_id, "pdf_path": pdf_path}, indent=2))


if __name__ == "__main__":
    main()
