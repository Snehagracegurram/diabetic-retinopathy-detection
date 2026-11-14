from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from PIL import Image

from code.config import LESION_CLASSES


DEFAULT_COLORS = {
    "microaneurysms": (255, 64, 64),
    "hemorrhages": (220, 20, 60),
    "hard_exudates": (255, 215, 0),
    "soft_exudates": (135, 206, 250),
    "optic_disc": (144, 238, 144),
}


def load_rgb(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def apply_clahe_rgb(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def normalize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4,
) -> np.ndarray:
    mask = (mask > 0.5).astype(np.uint8)
    colored = np.zeros_like(image)
    colored[:, :, 0] = color[0]
    colored[:, :, 1] = color[1]
    colored[:, :, 2] = color[2]
    out = image.copy()
    idx = mask > 0
    out[idx] = ((1 - alpha) * out[idx] + alpha * colored[idx]).astype(np.uint8)
    return out


def overlay_multiple_masks(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    out = image.copy()
    for name in LESION_CLASSES:
        if name in masks:
            out = overlay_mask(out, masks[name], DEFAULT_COLORS.get(name, (255, 255, 255)), alpha=alpha)
    return out


def draw_points(
    image: np.ndarray,
    points: Dict[str, Tuple[float, float]],
    radius: int = 8,
) -> np.ndarray:
    out = image.copy()
    for label, (x, y) in points.items():
        color = (0, 255, 0) if "optic" in label.lower() else (255, 0, 0)
        cv2.circle(out, (int(x), int(y)), radius, color, thickness=2)
        cv2.putText(
            out,
            label,
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended = ((1 - alpha) * image + alpha * colormap).astype(np.uint8)
    return blended


def image_to_base64(image: np.ndarray, format: str = "PNG") -> str:
    pil = Image.fromarray(image)
    buffer = BytesIO()
    pil.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def write_mask_png(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(path), mask)


def collate_visual_artifacts(
    output_dir: str | Path,
    image_name: str,
    input_image: np.ndarray,
    overlay: np.ndarray,
    gradcam: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / image_name
    save_rgb(base.with_name(base.name + "_input.png"), input_image)
    save_rgb(base.with_name(base.name + "_overlay.png"), overlay)
    save_rgb(base.with_name(base.name + "_gradcam.png"), gradcam)
    mask_paths: Dict[str, str] = {}
    for label, mask in masks.items():
        p = base.with_name(base.name + f"_mask_{label}.png")
        write_mask_png(p, mask)
        mask_paths[label] = str(p)
    return {
        "input": str(base.with_name(base.name + "_input.png")),
        "overlay": str(base.with_name(base.name + "_overlay.png")),
        "gradcam": str(base.with_name(base.name + "_gradcam.png")),
        **{f"mask_{k}": v for k, v in mask_paths.items()},
    }
