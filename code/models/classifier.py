from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B3_Weights,
    EfficientNet_B5_Weights,
    efficientnet_b3,
    efficientnet_b5,
)


def build_classifier(
    backbone: Literal["efficientnet_b3", "efficientnet_b5"] = "efficientnet_b3",
    num_classes: int = 5,
    dropout: float = 0.4,
    pretrained: bool = True,
) -> nn.Module:
    if backbone == "efficientnet_b3":
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[1].in_features
    elif backbone == "efficientnet_b5":
        model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unsupported classifier backbone: {backbone}")

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    # EfficientNet last feature block before pooling.
    return model.features[-1]


def load_classifier_checkpoint(path: str, device: torch.device) -> nn.Module:
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {})
    backbone = metadata.get("backbone", "efficientnet_b3")
    num_classes = metadata.get("num_classes", 5)
    model = build_classifier(backbone=backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
