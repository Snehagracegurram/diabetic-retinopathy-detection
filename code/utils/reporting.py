from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_pdf_report(
    output_path: str | Path,
    patient: Dict[str, str],
    inference: Dict[str, Any],
    artifact_paths: Dict[str, str],
    model_summary: Dict[str, Any],
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Diabetic Retinopathy AI Report", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    patient_table = Table(
        [
            ["Patient Name", patient.get("name", "")],
            ["Patient ID", patient.get("id", "")],
            ["Age", patient.get("age", "")],
            ["Sex", patient.get("sex", "")],
            ["Report Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ],
        colWidths=[4.5 * cm, 11 * cm],
    )
    patient_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(patient_table)
    story.append(Spacer(1, 0.4 * cm))

    pred = inference["classification"]
    story.append(
        Paragraph(
            f"Prediction: <b>{pred['label']}</b> (confidence {pred['confidence']:.3f})",
            styles["Heading3"],
        )
    )
    story.append(
        Paragraph(
            "This is a research/demo tool and is not cleared for clinical diagnosis.",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 0.3 * cm))
    loc = inference["localization"]
    loc_table = Table(
        [
            ["Structure", "X", "Y", "Uncertainty (px)"],
            [
                "Optic Disc",
                f"{loc['optic_disc']['x']:.1f}",
                f"{loc['optic_disc']['y']:.1f}",
                f"{loc['optic_disc']['uncertainty_px']:.2f}",
            ],
            [
                "Fovea",
                f"{loc['fovea']['x']:.1f}",
                f"{loc['fovea']['y']:.1f}",
                f"{loc['fovea']['uncertainty_px']:.2f}",
            ],
        ],
        colWidths=[5 * cm, 3 * cm, 3 * cm, 4.5 * cm],
    )
    loc_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    story.append(loc_table)
    story.append(Spacer(1, 0.4 * cm))

    for key, title in [
        ("input", "Input Fundus Image"),
        ("overlay", "Lesion Segmentation Overlay"),
        ("gradcam", "Grad-CAM Explainability"),
    ]:
        if key in artifact_paths and Path(artifact_paths[key]).exists():
            story.append(Paragraph(title, styles["Heading4"]))
            story.append(RLImage(artifact_paths[key], width=13.5 * cm, height=9 * cm))
            story.append(Spacer(1, 0.3 * cm))

    summary_rows = [["Model", "Summary"]]
    for k, v in model_summary.items():
        summary_rows.append([str(k), str(v)])
    sum_table = Table(summary_rows, colWidths=[5.5 * cm, 10 * cm])
    sum_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    story.append(Paragraph("Model & Training Summary", styles["Heading4"]))
    story.append(sum_table)

    doc.build(story)
    return str(output_path)
