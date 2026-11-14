from __future__ import annotations

import torch

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation-models-pytorch is required for segmentation model. Install via requirements.txt"
    ) from e


def build_segmentation_model(
    encoder_name: str = "efficientnet-b3",
    encoder_weights: str | None = "imagenet",
    classes: int = 5,
) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
        activation=None,
    )
    return model


def load_segmentation_checkpoint(path: str, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {})
    model = build_segmentation_model(
        encoder_name=metadata.get("encoder", "efficientnet-b3"),
        encoder_weights=None,
        classes=metadata.get("num_classes", 5),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
