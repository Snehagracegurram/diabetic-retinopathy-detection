from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from code.infer import DRInferencePipeline
from code.models.classifier import build_classifier, load_classifier_checkpoint
from code.models.localization import build_localization_model, load_localization_checkpoint
from code.models.segmentation import build_segmentation_model, load_segmentation_checkpoint


def _write_dummy_checkpoints(root: Path) -> None:
    cls_dir = root / "models" / "classification" / "single_split"
    seg_dir = root / "models" / "segmentation"
    loc_dir = root / "models" / "localization"
    cls_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    loc_dir.mkdir(parents=True, exist_ok=True)

    classifier = build_classifier(backbone="efficientnet_b3", num_classes=5, pretrained=False)
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "metadata": {"backbone": "efficientnet_b3", "num_classes": 5},
        },
        cls_dir / "best.pt",
    )

    segmenter = build_segmentation_model(encoder_name="efficientnet-b3", encoder_weights=None, classes=5)
    torch.save(
        {
            "model_state_dict": segmenter.state_dict(),
            "metadata": {"encoder": "efficientnet-b3", "num_classes": 5},
        },
        seg_dir / "best.pt",
    )

    localizer = build_localization_model(pretrained=False)
    torch.save(
        {
            "model_state_dict": localizer.state_dict(),
            "metadata": {"model_name": "resnet18_localization"},
        },
        loc_dir / "best.pt",
    )

    (root / "models" / "classification" / "metadata.json").write_text(
        '{"model_name":"dr_classifier","best_metric":0.0,"date":"test"}', encoding="utf-8"
    )
    (root / "models" / "segmentation" / "metadata.json").write_text(
        '{"model_name":"dr_segmenter","best_metric":0.0,"date":"test"}', encoding="utf-8"
    )
    (root / "models" / "localization" / "metadata.json").write_text(
        '{"model_name":"dr_localizer","best_metric":0.0,"date":"test"}', encoding="utf-8"
    )


def test_model_loaders(tmp_path: Path):
    _write_dummy_checkpoints(tmp_path)
    device = torch.device("cpu")

    cls_model = load_classifier_checkpoint(str(tmp_path / "models/classification/single_split/best.pt"), device)
    seg_model = load_segmentation_checkpoint(str(tmp_path / "models/segmentation/best.pt"), device)
    loc_model = load_localization_checkpoint(str(tmp_path / "models/localization/best.pt"), device)

    assert cls_model is not None
    assert seg_model is not None
    assert loc_model is not None


def test_inference_pipeline_smoke(tmp_path: Path):
    _write_dummy_checkpoints(tmp_path)

    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 180, (120, 80, 200), thickness=-1)
    input_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(input_path), img)

    pipeline = DRInferencePipeline(project_root=tmp_path, force_device="cpu")
    result = pipeline.infer(
        image_path=input_path,
        output_dir=tmp_path / "results",
        patient={"name": "Unit Test", "id": "UT-01", "age": "30", "sex": "Other"},
        db=None,
        generate_pdf=False,
    )

    assert "classification" in result
    assert "segmentation" in result
    assert "localization" in result
    assert "gradcam" in result
    assert set(result["segmentation"].keys()) >= {
        "microaneurysms",
        "hemorrhages",
        "hard_exudates",
        "soft_exudates",
        "optic_disc",
    }
