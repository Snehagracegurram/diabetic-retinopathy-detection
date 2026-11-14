from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from code.config import CLASS_NAMES, LESION_CLASSES
from code.models.classifier import get_gradcam_target_layer, load_classifier_checkpoint
from code.models.localization import load_localization_checkpoint
from code.models.segmentation import load_segmentation_checkpoint
from code.utils.database import InferenceDB
from code.utils.gradcam import GradCAM
from code.utils.reporting import generate_pdf_report
from code.utils.system import utc_now_iso
from code.utils.visualization import (
    collate_visual_artifacts,
    draw_points,
    heatmap_overlay,
    image_to_base64,
    load_rgb,
    overlay_multiple_masks,
)


def _build_infer_transform(size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _find_checkpoint(base_dir: Path) -> Optional[Path]:
    if base_dir.is_file() and base_dir.suffix == ".pt":
        return base_dir
    candidates = list(base_dir.rglob("best.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_metadata(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class DRInferencePipeline:
    def __init__(
        self,
        project_root: str | Path = ".",
        classifier_ckpt: str | Path = "models/classification",
        segmentation_ckpt: str | Path = "models/segmentation",
        localization_ckpt: str | Path = "models/localization",
        image_size: int = 512,
        force_device: str = "cuda",
    ):
        self.project_root = Path(project_root)
        self.image_size = image_size
        if force_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU REQUIRED -- aborting. CUDA device not available for inference. "
                "Use --device cpu to run CPU inference or install CUDA-enabled PyTorch."
            )
        self.device = torch.device(force_device if force_device else ("cuda" if torch.cuda.is_available() else "cpu"))

        cls_ckpt = _find_checkpoint(self.project_root / classifier_ckpt)
        seg_ckpt = _find_checkpoint(self.project_root / segmentation_ckpt)
        loc_ckpt = _find_checkpoint(self.project_root / localization_ckpt)

        missing = []
        if cls_ckpt is None:
            missing.append("classification checkpoint")
        if seg_ckpt is None:
            missing.append("segmentation checkpoint")
        if loc_ckpt is None:
            missing.append("localization checkpoint")
        if missing:
            raise FileNotFoundError(
                "Model checkpoint(s) missing. Please train models first: " + ", ".join(missing)
            )

        self.classifier = load_classifier_checkpoint(str(cls_ckpt), self.device)
        self.segmenter = load_segmentation_checkpoint(str(seg_ckpt), self.device)
        self.localizer = load_localization_checkpoint(str(loc_ckpt), self.device)

        self.gradcam = GradCAM(self.classifier, get_gradcam_target_layer(self.classifier))
        self.transform = _build_infer_transform(image_size)

        self.classifier_meta = _read_metadata((self.project_root / "models/classification/metadata.json"))
        self.segmentation_meta = _read_metadata((self.project_root / "models/segmentation/metadata.json"))
        self.localization_meta = _read_metadata((self.project_root / "models/localization/metadata.json"))

        self.version = {
            "classifier": self.classifier_meta.get("date", "unknown"),
            "segmenter": self.segmentation_meta.get("date", "unknown"),
            "localizer": self.localization_meta.get("date", "unknown"),
        }

    def _prepare(self, image_rgb: np.ndarray) -> torch.Tensor:
        out = self.transform(image=image_rgb)
        x = out["image"].unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def _predict_classification(self, x: torch.Tensor) -> Dict[str, Any]:
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        return {
            "class_index": idx,
            "label": CLASS_NAMES[idx],
            "confidence": conf,
            "probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))},
        }

    @torch.no_grad()
    def _predict_segmentation(self, x: torch.Tensor, original_shape: tuple[int, int]) -> Dict[str, np.ndarray]:
        logits = self.segmenter(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        h, w = original_shape
        masks = {}
        for i, cls in enumerate(LESION_CLASSES):
            masks[cls] = cv2.resize(probs[i], (w, h), interpolation=cv2.INTER_LINEAR)
        return masks

    @torch.no_grad()
    def _predict_localization(self, x: torch.Tensor, original_shape: tuple[int, int]) -> Dict[str, Any]:
        pred_native = self.localizer(x)[0].detach().cpu().numpy()
        pred_flip = self.localizer(torch.flip(x, dims=[3]))[0].detach().cpu().numpy()

        # The localizer is trained in a canonical orientation.
        # Evaluate both original and mirrored inputs, then map the mirrored
        # prediction back to the original frame and select the more plausible one.
        pred_flip[0] = 1.0 - pred_flip[0]
        pred_flip[2] = 1.0 - pred_flip[2]

        def _plausibility_score(pred_arr: np.ndarray) -> float:
            od_x, od_y, fv_x, fv_y = [float(v) for v in pred_arr[:4]]
            unc = float(max(pred_arr[4], 0.0) + max(pred_arr[5], 0.0))
            dist = float(np.hypot(od_x - fv_x, od_y - fv_y))
            penalty = 0.0
            if dist < 0.12:
                penalty += (0.12 - dist) * 2.0
            if dist > 0.60:
                penalty += (dist - 0.60) * 2.0
            return unc + penalty

        pred = pred_native if _plausibility_score(pred_native) <= _plausibility_score(pred_flip) else pred_flip
        pred[:4] = np.clip(pred[:4], 0.0, 1.0)
        h, w = original_shape

        od_x = float(pred[0] * w)
        od_y = float(pred[1] * h)
        fv_x = float(pred[2] * w)
        fv_y = float(pred[3] * h)

        # Convert uncertainty outputs to pixel-level estimate.
        od_unc = float(pred[4] * max(h, w))
        fv_unc = float(pred[5] * max(h, w))

        return {
            "optic_disc": {"x": od_x, "y": od_y, "uncertainty_px": od_unc},
            "fovea": {"x": fv_x, "y": fv_y, "uncertainty_px": fv_unc},
        }

    def infer(
        self,
        image_path: str | Path,
        output_dir: str | Path = "results/inference",
        patient: Optional[Dict[str, str]] = None,
        db: Optional[InferenceDB] = None,
        generate_pdf: bool = True,
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        patient = patient or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_rgb = load_rgb(image_path)
        h, w = image_rgb.shape[:2]

        x = self._prepare(image_rgb)

        classification = self._predict_classification(x)
        segmentation_masks = self._predict_segmentation(x, original_shape=(h, w))
        localization = self._predict_localization(x, original_shape=(h, w))

        cam = self.gradcam(x, class_idx=classification["class_index"]) 
        gradcam_map = cv2.resize(cam.heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        overlay = overlay_multiple_masks(image_rgb, segmentation_masks, alpha=0.45)
        points_img = draw_points(
            overlay,
            {
                "optic_disc": (localization["optic_disc"]["x"], localization["optic_disc"]["y"]),
                "fovea": (localization["fovea"]["x"], localization["fovea"]["y"]),
            },
        )
        gradcam_overlay = heatmap_overlay(image_rgb, gradcam_map, alpha=0.45)

        artifacts = collate_visual_artifacts(
            output_dir=output_dir,
            image_name=image_path.stem,
            input_image=image_rgb,
            overlay=points_img,
            gradcam=gradcam_overlay,
            masks=segmentation_masks,
        )

        result = {
            "timestamp_utc": utc_now_iso(),
            "input_image": str(image_path),
            "classification": classification,
            "segmentation": {
                cls: {
                    "mask_base64": image_to_base64((mask > 0.5).astype(np.uint8) * 255),
                    "mean_probability": float(mask.mean()),
                }
                for cls, mask in segmentation_masks.items()
            },
            "localization": localization,
            "gradcam": {
                "class_index": cam.class_index,
                "score": cam.score,
                "image_base64": image_to_base64(gradcam_overlay),
            },
            "visuals": {
                "input_base64": image_to_base64(image_rgb),
                "segmentation_overlay_base64": image_to_base64(overlay),
                "localization_overlay_base64": image_to_base64(points_img),
            },
            "artifacts": artifacts,
            "model_version": self.version,
            "pdf_generated": False,
        }

        inference_id = None
        if db is not None:
            inference_id = db.insert_inference(
                created_at=result["timestamp_utc"],
                patient=patient,
                input_image_path=str(image_path),
                prediction=result,
                pdf_path=None,
            )
            result["inference_id"] = inference_id

        if generate_pdf:
            report_dir = output_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{image_path.stem}_{result['timestamp_utc'].replace(':', '').replace('-', '')}.pdf"
            pdf_path = generate_pdf_report(
                output_path=report_path,
                patient=patient,
                inference=result,
                artifact_paths=artifacts,
                model_summary={
                    "classification_model": self.classifier_meta.get("model_name", "dr_classifier"),
                    "classification_best_metric": self.classifier_meta.get("best_metric", "n/a"),
                    "segmentation_model": self.segmentation_meta.get("model_name", "dr_segmenter"),
                    "segmentation_best_metric": self.segmentation_meta.get("best_metric", "n/a"),
                    "localization_model": self.localization_meta.get("model_name", "dr_localizer"),
                    "localization_best_metric": self.localization_meta.get("best_metric", "n/a"),
                },
            )
            result["report_pdf"] = str(pdf_path)
            result["pdf_generated"] = True
            if db is not None and inference_id is not None:
                db.update_pdf_path(inference_id, str(pdf_path))

        return result



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image DR inference pipeline.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/inference")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patient-name", type=str, default="")
    parser.add_argument("--patient-id", type=str, default="")
    parser.add_argument("--patient-age", type=str, default="")
    parser.add_argument("--patient-sex", type=str, default="")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--db-path", type=str, default="app/dr_inference.db")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = DRInferencePipeline(
        project_root=args.project_root,
        image_size=args.image_size,
        force_device=args.device,
    )
    db = InferenceDB(Path(args.project_root) / args.db_path)

    result = pipeline.infer(
        image_path=args.input,
        output_dir=Path(args.project_root) / args.out,
        patient={
            "name": args.patient_name,
            "id": args.patient_id,
            "age": args.patient_age,
            "sex": args.patient_sex,
        },
        db=db,
        generate_pdf=not args.no_pdf,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
