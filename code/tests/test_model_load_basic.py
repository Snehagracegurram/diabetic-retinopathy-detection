from __future__ import annotations

from pathlib import Path

import torch

from code.models.classifier import build_classifier, load_classifier_checkpoint
from code.models.localization import build_localization_model, load_localization_checkpoint


def test_basic_model_loading(tmp_path: Path):
    cls = build_classifier(backbone="efficientnet_b3", num_classes=5, pretrained=False)
    loc = build_localization_model(pretrained=False)

    cls_path = tmp_path / "cls.pt"
    loc_path = tmp_path / "loc.pt"

    torch.save({"model_state_dict": cls.state_dict(), "metadata": {"backbone": "efficientnet_b3", "num_classes": 5}}, cls_path)
    torch.save({"model_state_dict": loc.state_dict(), "metadata": {}}, loc_path)

    device = torch.device("cpu")
    loaded_cls = load_classifier_checkpoint(str(cls_path), device)
    loaded_loc = load_localization_checkpoint(str(loc_path), device)

    assert loaded_cls is not None
    assert loaded_loc is not None
