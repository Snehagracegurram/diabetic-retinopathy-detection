from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from code.utils.database import InferenceDB


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB


def create_app(
    project_root: str | Path = ".",
    pipeline_override=None,
    db_path_override: str | Path | None = None,
) -> Flask:
    project_root = Path(project_root).resolve()
    frontend_dir = project_root / "code" / "frontend"
    app_dir = project_root / "app"
    upload_dir = app_dir / "uploads"
    inference_dir = project_root / "results" / "inference"

    app = Flask(
        __name__,
        static_folder=str(frontend_dir),
        template_folder=str(frontend_dir),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = str(upload_dir)

    upload_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(db_path_override) if db_path_override is not None else (app_dir / "dr_inference.db")
    db = InferenceDB(db_path)

    pipeline = None
    pipeline_error = None
    if pipeline_override is not None:
        pipeline = pipeline_override
    else:
        try:
            from code.infer import DRInferencePipeline

            pipeline = DRInferencePipeline(project_root=project_root)
        except Exception as e:
            pipeline_error = str(e)

    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/assets/<path:filename>")
    def assets(filename: str):
        return send_from_directory(frontend_dir / "assets", filename)

    @app.route("/api/health")
    def health():
        if pipeline is None:
            return jsonify(
                {
                    "status": "degraded",
                    "message": "Models not loaded",
                    "error": pipeline_error,
                }
            ), 503
        return jsonify({"status": "ok", "model_version": pipeline.version})

    @app.route("/api/infer", methods=["POST"])
    def infer_route():
        nonlocal pipeline
        if pipeline is None:
            return jsonify({"error": "Model pipeline unavailable", "details": pipeline_error}), 503

        if "image" not in request.files:
            return jsonify({"error": "Missing image file field 'image'"}), 400

        file = request.files["image"]
        if file.filename is None or file.filename.strip() == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file extension: {suffix}"}), 400

        save_path = upload_dir / filename
        file.save(save_path)

        patient = {
            "name": request.form.get("patient_name", ""),
            "id": request.form.get("patient_id", ""),
            "age": request.form.get("patient_age", ""),
            "sex": request.form.get("patient_sex", ""),
        }

        try:
            result = pipeline.infer(
                image_path=save_path,
                output_dir=inference_dir,
                patient=patient,
                db=db,
                generate_pdf=True,
            )
            return jsonify({"status": "ok", "result": result})
        except Exception as e:
            return jsonify({"error": "Inference failed", "details": str(e)}), 500

    @app.route("/api/report/<int:inference_id>", methods=["GET"])
    def report_route(inference_id: int):
        row = db.get_inference(inference_id)
        if row is None:
            return jsonify({"error": "Report not found"}), 404
        pdf_path = row.get("pdf_path")
        if not pdf_path:
            return jsonify({"error": "PDF not generated for this inference"}), 404
        pdf = Path(pdf_path)
        if not pdf.exists():
            return jsonify({"error": "PDF file missing from disk"}), 404
        return send_file(pdf, as_attachment=True, download_name=pdf.name)

    return app


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    app = create_app(project_root)
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
