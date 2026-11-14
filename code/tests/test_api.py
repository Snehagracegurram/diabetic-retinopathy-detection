from __future__ import annotations

from pathlib import Path

from code.backend.app import create_app


class FakePipeline:
    def __init__(self):
        self.version = {"classifier": "test", "segmenter": "test", "localizer": "test"}

    def infer(self, image_path, output_dir, patient, db, generate_pdf=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir = output_dir / "reports"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "fake_report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%Fake\n")

        result = {
            "classification": {"label": "No DR", "confidence": 0.99, "probabilities": {"No DR": 0.99}},
            "segmentation": {},
            "localization": {
                "optic_disc": {"x": 10.0, "y": 20.0, "uncertainty_px": 2.0},
                "fovea": {"x": 30.0, "y": 40.0, "uncertainty_px": 3.0},
            },
            "gradcam": {"image_base64": ""},
            "report_pdf": str(pdf_path),
            "model_version": self.version,
        }

        inference_id = db.insert_inference(
            created_at="2026-01-01T00:00:00Z",
            patient=patient,
            input_image_path=str(image_path),
            prediction=result,
            pdf_path=str(pdf_path),
        )
        result["inference_id"] = inference_id
        return result


def test_health_and_infer_endpoints(tmp_path: Path):
    app = create_app(project_root=tmp_path, pipeline_override=FakePipeline(), db_path_override=tmp_path / "test.db")
    client = app.test_client()

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json["status"] == "ok"

    # Missing file validation.
    bad_resp = client.post("/api/infer", data={})
    assert bad_resp.status_code == 400

    img_path = tmp_path / "x.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xd9")

    with img_path.open("rb") as f:
        resp = client.post(
            "/api/infer",
            data={
                "image": (f, "x.jpg"),
                "patient_name": "Alice",
                "patient_id": "P1",
                "patient_age": "45",
                "patient_sex": "Female",
            },
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    payload = resp.json
    assert payload["status"] == "ok"
    assert payload["result"]["inference_id"] > 0

    report_resp = client.get(f"/api/report/{payload['result']['inference_id']}")
    assert report_resp.status_code == 200
    assert report_resp.mimetype == "application/pdf"
