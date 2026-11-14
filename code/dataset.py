from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from code.config import LESION_CLASSES
from code.utils.system import append_run_summary


@dataclass
class DatasetPaths:
    aptos: Path
    idrid: Path
    ddr: Path


CLASS_NAME_TO_DDR_CODE = {
    "No DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferative DR": 4,
}

SEG_CLASS_TO_DDR_DIR = {
    "microaneurysms": "MA",
    "hemorrhages": "HE",
    "hard_exudates": "EX",
    "soft_exudates": "SE",
}


def _find_dir_with_predicate(root: Path, predicate) -> Optional[Path]:
    for p in root.rglob("*"):
        if p.is_dir() and predicate(p):
            return p
    return None


def discover_datasets(
    project_root: str | Path = ".",
    raw_root: str | Path = "data/raw",
    strict: bool = True,
) -> DatasetPaths:
    project_root = Path(project_root).resolve()
    raw_root = (project_root / raw_root).resolve()

    expected = {
        "aptos": raw_root / "aptos2019",
        "idrid": raw_root / "IDRiD",
        "ddr": raw_root / "DDR",
    }

    aptos_path = expected["aptos"] if expected["aptos"].exists() else None
    idrid_path = expected["idrid"] if expected["idrid"].exists() else None
    ddr_path = expected["ddr"] if expected["ddr"].exists() else None

    # Auto-discovery fallback when canonical paths are absent.
    if aptos_path is None:
        aptos_path = _find_dir_with_predicate(
            project_root,
            lambda p: (p / "train.csv").exists() and (p / "train_images").exists(),
        )
    if idrid_path is None:
        idrid_path = _find_dir_with_predicate(
            project_root,
            lambda p: (p / "A. Segmentation").exists() and (p / "C. Localization").exists(),
        )
    if ddr_path is None:
        ddr_path = _find_dir_with_predicate(
            project_root,
            lambda p: (p / "DR_grading").exists() and (p / "lesion_segmentation").exists(),
        )

    missing = []
    if aptos_path is None:
        missing.append("data/raw/aptos2019")
    if idrid_path is None:
        missing.append("data/raw/IDRiD")
    if ddr_path is None:
        missing.append("data/raw/DDR")

    if strict and missing:
        raise FileNotFoundError(
            "Missing required datasets:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\nPlace datasets under data/raw/ or ensure auto-discovery paths exist locally."
        )

    return DatasetPaths(
        aptos=aptos_path if aptos_path is not None else expected["aptos"],
        idrid=idrid_path if idrid_path is not None else expected["idrid"],
        ddr=ddr_path if ddr_path is not None else expected["ddr"],
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def validate_checksums(root: str | Path, manifest_path: str | Path) -> Dict[str, str]:
    root = Path(root)
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    mismatched = {}
    for rel_path, expected_sha in manifest.items():
        full_path = root / rel_path
        if not full_path.exists():
            mismatched[rel_path] = "missing"
            continue
        current = _sha256(full_path)
        if current.lower() != str(expected_sha).lower():
            mismatched[rel_path] = current
    return mismatched


def _clean_numeric(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("nan")


def _ensure_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def load_aptos_dataframe(aptos_root: Path) -> pd.DataFrame:
    csv_path = aptos_root / "train.csv"
    img_root = aptos_root / "train_images"
    if not csv_path.exists():
        return pd.DataFrame(columns=["image_id", "image_path", "label", "source"]) 

    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        image_id = str(row["id_code"])
        label = int(row["diagnosis"])
        image_path = img_root / f"{image_id}.png"
        if not _ensure_exists(image_path):
            continue
        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "label": label,
                "source": "aptos2019",
            }
        )
    return pd.DataFrame(records)


def _load_ddr_split(ddr_root: Path, split: str) -> pd.DataFrame:
    split_file = ddr_root / "DR_grading" / f"{split}.txt"
    image_root = ddr_root / "DR_grading" / split
    if not split_file.exists() or not image_root.exists():
        return pd.DataFrame(columns=["image_id", "image_path", "label", "source"]) 

    records = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            image_name, label_s = parts[0], parts[1]
            image_path = image_root / image_name
            if not _ensure_exists(image_path):
                continue
            records.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "label": int(label_s),
                    "source": f"ddr_{split}",
                }
            )
    return pd.DataFrame(records)


def load_ddr_grading_dataframe(ddr_root: Path) -> pd.DataFrame:
    frames = [_load_ddr_split(ddr_root, s) for s in ["train", "valid", "test"]]
    if not frames:
        return pd.DataFrame(columns=["image_id", "image_path", "label", "source"]) 
    return pd.concat(frames, ignore_index=True)


def _load_idrid_grade_csv(csv_path: Path, image_root: Path, source: str) -> pd.DataFrame:
    if not csv_path.exists() or not image_root.exists():
        return pd.DataFrame(columns=["image_id", "image_path", "label", "source"]) 

    df = pd.read_csv(csv_path)
    img_col = next((c for c in df.columns if "Image" in c), df.columns[0])
    grade_col = next((c for c in df.columns if "Retinopathy grade" in c), df.columns[1])

    records = []
    for _, row in df.iterrows():
        image_id = str(row[img_col]).strip()
        label = int(_clean_numeric(row[grade_col]))
        image_path = image_root / f"{image_id}.jpg"
        if not _ensure_exists(image_path):
            alt = image_root / f"{image_id}.png"
            if _ensure_exists(alt):
                image_path = alt
            else:
                continue
        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "label": label,
                "source": source,
            }
        )
    return pd.DataFrame(records)


def load_idrid_grading_dataframe(idrid_root: Path) -> pd.DataFrame:
    grade_root = idrid_root / "B. Disease Grading"
    train_csv = grade_root / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv = grade_root / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
    train_img_root = grade_root / "1. Original Images" / "a. Training Set"
    test_img_root = grade_root / "1. Original Images" / "b. Testing Set"

    train_df = _load_idrid_grade_csv(train_csv, train_img_root, "idrid_train")
    test_df = _load_idrid_grade_csv(test_csv, test_img_root, "idrid_test")
    return pd.concat([train_df, test_df], ignore_index=True)


def build_classification_dataframe(paths: DatasetPaths) -> pd.DataFrame:
    frames = [
        load_aptos_dataframe(paths.aptos),
        load_ddr_grading_dataframe(paths.ddr),
        load_idrid_grading_dataframe(paths.idrid),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].between(0, 4)].reset_index(drop=True)
    return df


def save_classification_splits(
    df: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
    folds: int = 5,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val["label"],
    )

    paths = {
        "train": output_dir / "classification_train.csv",
        "val": output_dir / "classification_val.csv",
        "test": output_dir / "classification_test.csv",
    }
    train.to_csv(paths["train"], index=False)
    val.to_csv(paths["val"], index=False)
    test.to_csv(paths["test"], index=False)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_records = train_val.reset_index(drop=True)
    for i, (tr_idx, va_idx) in enumerate(skf.split(fold_records, fold_records["label"])):
        tr_df = fold_records.iloc[tr_idx]
        va_df = fold_records.iloc[va_idx]
        tr_df.to_csv(output_dir / f"classification_fold{i}_train.csv", index=False)
        va_df.to_csv(output_dir / f"classification_fold{i}_val.csv", index=False)

    return paths


def _read_coord_csv(path: Path) -> Dict[str, Tuple[float, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = list(df.columns)
    img_col = next((c for c in cols if "Image" in c), cols[0])
    x_col = next((c for c in cols if "X" in c), cols[1])
    y_col = next((c for c in cols if "Y" in c), cols[2])
    out = {}
    for _, row in df.iterrows():
        image_id = str(row[img_col]).strip()
        x = _clean_numeric(row[x_col])
        y = _clean_numeric(row[y_col])
        if np.isnan(x) or np.isnan(y):
            continue
        out[image_id] = (float(x), float(y))
    return out


def load_localization_dataframe(idrid_root: Path) -> pd.DataFrame:
    loc_root = idrid_root / "C. Localization"
    od_train = loc_root / "2. Groundtruths" / "1. Optic Disc Center Location" / "a. IDRiD_OD_Center_Training Set_Markups.csv"
    od_test = loc_root / "2. Groundtruths" / "1. Optic Disc Center Location" / "b. IDRiD_OD_Center_Testing Set_Markups.csv"
    fv_train = loc_root / "2. Groundtruths" / "2. Fovea Center Location" / "IDRiD_Fovea_Center_Training Set_Markups.csv"
    fv_test = loc_root / "2. Groundtruths" / "2. Fovea Center Location" / "IDRiD_Fovea_Center_Testing Set_Markups.csv"
    img_train_root = loc_root / "1. Original Images" / "a. Training Set"
    img_test_root = loc_root / "1. Original Images" / "b. Testing Set"

    od = {}
    od.update(_read_coord_csv(od_train))
    od.update(_read_coord_csv(od_test))
    fv = {}
    fv.update(_read_coord_csv(fv_train))
    fv.update(_read_coord_csv(fv_test))

    records = []
    for image_id in sorted(set(od.keys()) & set(fv.keys())):
        image_path = img_train_root / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = img_test_root / f"{image_id}.jpg"
        if not image_path.exists():
            continue
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        odx, ody = od[image_id]
        fvx, fvy = fv[image_id]
        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "width": w,
                "height": h,
                "od_x": odx,
                "od_y": ody,
                "fv_x": fvx,
                "fv_y": fvy,
            }
        )
    return pd.DataFrame(records)


def _build_optic_disc_mask(shape: Tuple[int, int], center_xy: Tuple[float, float], radius_frac: float = 0.04) -> np.ndarray:
    h, w = shape
    radius = max(6, int(min(h, w) * radius_frac))
    cx, cy = int(center_xy[0]), int(center_xy[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, color=255, thickness=-1)
    return mask


def _find_idrid_segmentation_masks(idrid_root: Path) -> Dict[str, Dict[str, Path]]:
    # Support multiple possible IDRiD extractions by searching file names.
    class_key_patterns = {
        "microaneurysms": ["micro", "ma"],
        "hemorrhages": ["haem", "hem"],
        "hard_exudates": ["hard", "exudate"],
        "soft_exudates": ["soft", "cotton", "se"],
    }
    id_to_paths: Dict[str, Dict[str, Path]] = {}

    for path in idrid_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".tif", ".png", ".jpg", ".bmp"}:
            continue
        stem = path.stem
        if not stem.startswith("IDRiD_"):
            continue
        lower_path = str(path).lower()
        matched_class = None
        for cls, patterns in class_key_patterns.items():
            if any(p in lower_path for p in patterns):
                matched_class = cls
                break
        if matched_class is None:
            continue
        id_to_paths.setdefault(stem[:9], {})[matched_class] = path

    return id_to_paths


def load_ddr_segmentation_records(ddr_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for split in ["train", "valid", "test"]:
        split_root = ddr_root / "lesion_segmentation" / split
        img_root = split_root / "image"
        lbl_root = split_root / "label"
        if not img_root.exists() or not lbl_root.exists():
            continue

        for img_path in sorted(img_root.glob("*.jpg")) + sorted(img_root.glob("*.png")) + sorted(img_root.glob("*.tif")):
            stem = img_path.stem
            mask_paths = {}
            for cls, ddr_tag in SEG_CLASS_TO_DDR_DIR.items():
                candidate = lbl_root / ddr_tag / f"{stem}.tif"
                if candidate.exists():
                    mask_paths[cls] = str(candidate)
            records.append(
                {
                    "image_id": stem,
                    "image_path": str(img_path),
                    "source": f"ddr_seg_{split}",
                    "mask_paths": mask_paths,
                }
            )
    return records


def load_segmentation_records(paths: DatasetPaths, run_summary_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    loc_df = load_localization_dataframe(paths.idrid)
    loc_lookup = {
        row.image_id: (float(row.od_x), float(row.od_y), int(row.width), int(row.height))
        for row in loc_df.itertuples(index=False)
    }

    idrid_masks = _find_idrid_segmentation_masks(paths.idrid)

    idrid_img_roots = [
        paths.idrid / "A. Segmentation" / "1. Original Images" / "a. Training Set",
        paths.idrid / "A. Segmentation" / "1. Original Images" / "b. Testing Set",
    ]
    for img_root in idrid_img_roots:
        if not img_root.exists():
            continue
        for img_path in sorted(img_root.glob("IDRiD_*.jpg")):
            image_id = img_path.stem
            mask_paths = {k: str(v) for k, v in idrid_masks.get(image_id, {}).items()}
            if image_id in loc_lookup:
                odx, ody, w, h = loc_lookup[image_id]
                mask_paths["optic_disc_center"] = (odx, ody)
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(img_path),
                    "source": "idrid_seg",
                    "mask_paths": mask_paths,
                }
            )

    records.extend(load_ddr_segmentation_records(paths.ddr))

    mismatches = 0
    valid_count = 0
    for rec in records:
        if Path(rec["image_path"]).exists():
            valid_count += 1
        else:
            mismatches += 1
    if run_summary_path is not None:
        append_run_summary(
            run_summary_path,
            stage="segmentation_record_validation",
            status="ok",
            details={
                "records_total": len(records),
                "records_with_existing_images": valid_count,
                "missing_image_records": mismatches,
            },
        )
    return records


def split_segmentation_records(
    records: List[Dict[str, Any]],
    out_dir: str | Path,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        train = []
        val = []
        test = []
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(records))
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        n_test = min(max(n_test, 1), max(n - 2, 1))
        n_val = min(max(n_val, 1), max(n - n_test - 1, 1))
        tr_end = n - (n_val + n_test)
        tr_idx = idx[:tr_end]
        va_idx = idx[tr_end : tr_end + n_val]
        te_idx = idx[tr_end + n_val :]
        train = [records[i] for i in tr_idx]
        val = [records[i] for i in va_idx]
        test = [records[i] for i in te_idx]

    train_path = out_dir / "segmentation_train.json"
    val_path = out_dir / "segmentation_val.json"
    test_path = out_dir / "segmentation_test.json"
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)

    return {"train": train_path, "val": val_path, "test": test_path}


def split_localization_dataframe(
    df: pd.DataFrame,
    out_dir: str | Path,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        train_df = df.copy()
        val_df = df.copy()
        test_df = df.copy()
    else:
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        rel_val = val_size / max(1e-6, 1.0 - test_size)
        train_df, val_df = train_test_split(train_val_df, test_size=rel_val, random_state=seed)
    train_path = out_dir / "localization_train.csv"
    val_path = out_dir / "localization_val.csv"
    test_path = out_dir / "localization_test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {"train": train_path, "val": val_path, "test": test_path}


def get_classification_transforms(image_size: int, train: bool = True) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.4),
                A.CLAHE(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_segmentation_transforms(image_size: int, train: bool = True) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.4),
                A.CLAHE(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_localization_transforms(image_size: int, train: bool = True) -> A.Compose:
    base = [A.Resize(image_size, image_size)]
    if train:
        # Keep localization augmentation photometric-only to avoid destabilizing
        # geometric landmarks on this small dataset.
        base.extend(
            [
                A.RandomBrightnessContrast(p=0.4),
                A.CLAHE(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            ]
        )
    base.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(
        base,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class ClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int, train: bool):
        self.df = dataframe.reset_index(drop=True)
        self.transform = get_classification_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = self.transform(image=image)
        return {
            "image": out["image"],
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_id": row["image_id"],
        }


class SegmentationDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], image_size: int, train: bool):
        self.records = records
        self.transform = get_segmentation_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.records)

    def _load_mask(self, path: Optional[str], shape: Tuple[int, int]) -> np.ndarray:
        if path is None:
            return np.zeros(shape, dtype=np.uint8)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros(shape, dtype=np.uint8)
        return mask

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing image: {rec['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask_paths = rec.get("mask_paths", {})
        masks = []
        for cls in LESION_CLASSES:
            if cls == "optic_disc":
                if "optic_disc" in mask_paths and isinstance(mask_paths["optic_disc"], str):
                    mask = self._load_mask(mask_paths["optic_disc"], (h, w))
                elif "optic_disc_center" in mask_paths:
                    odx, ody = mask_paths["optic_disc_center"]
                    mask = _build_optic_disc_mask((h, w), (odx, ody))
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)
            else:
                mask = self._load_mask(mask_paths.get(cls), (h, w))
            masks.append(mask)

        mask_stack = np.stack(masks, axis=-1)
        aug = self.transform(image=image, mask=mask_stack)
        image_t = aug["image"]
        mask_t = aug["mask"].permute(2, 0, 1).float() / 255.0

        return {
            "image": image_t,
            "mask": mask_t,
            "image_id": rec.get("image_id", Path(rec["image_path"]).stem),
        }


class LocalizationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int, train: bool):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = int(image_size)
        self.transform = get_localization_transforms(image_size=image_size, train=train)
        # Canonicalize orientation so optic disc appears on a consistent side.
        self.canonicalize_by_od = True

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        odx = float(row["od_x"])
        ody = float(row["od_y"])
        fvx = float(row["fv_x"])
        fvy = float(row["fv_y"])

        target_orig = np.array(
            [
                odx / max(w, 1),
                ody / max(h, 1),
                fvx / max(w, 1),
                fvy / max(h, 1),
            ],
            dtype=np.float32,
        )

        hflip_applied = 0.0
        if self.canonicalize_by_od and odx > (w * 0.5):
            image = cv2.flip(image, 1)
            odx = (w - 1.0) - odx
            fvx = (w - 1.0) - fvx
            hflip_applied = 1.0

        keypoints = [
            (odx, ody),
            (fvx, fvy),
        ]
        aug = self.transform(image=image, keypoints=keypoints)
        image_t = aug["image"]
        aug_keypoints = aug["keypoints"]
        if len(aug_keypoints) != 2:
            raise RuntimeError(f"Unexpected keypoint count after augmentation: {len(aug_keypoints)}")

        out_h, out_w = int(image_t.shape[1]), int(image_t.shape[2])
        (odx, ody), (fvx, fvy) = aug_keypoints
        odx = float(np.clip(odx, 0.0, out_w - 1.0))
        ody = float(np.clip(ody, 0.0, out_h - 1.0))
        fvx = float(np.clip(fvx, 0.0, out_w - 1.0))
        fvy = float(np.clip(fvy, 0.0, out_h - 1.0))

        target = np.array(
            [
                odx / max(out_w, 1),
                ody / max(out_h, 1),
                fvx / max(out_w, 1),
                fvy / max(out_h, 1),
            ],
            dtype=np.float32,
        )
        return {
            "image": image_t,
            "target": torch.tensor(target, dtype=torch.float32),
            "target_orig": torch.tensor(target_orig, dtype=torch.float32),
            "image_id": row["image_id"],
            "orig_size": torch.tensor([w, h], dtype=torch.float32),
            "hflip_applied": torch.tensor(hflip_applied, dtype=torch.float32),
        }


def validate_segmentation_pairing(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_by_class = {k: 0 for k in LESION_CLASSES}
    total = 0
    skipped = 0

    for rec in records:
        total += 1
        img_exists = Path(rec["image_path"]).exists()
        if not img_exists:
            skipped += 1
            continue
        mask_paths = rec.get("mask_paths", {})
        for cls in LESION_CLASSES:
            if cls == "optic_disc":
                has = ("optic_disc" in mask_paths and Path(mask_paths["optic_disc"]).exists()) or (
                    "optic_disc_center" in mask_paths
                )
            else:
                p = mask_paths.get(cls)
                has = p is not None and Path(p).exists()
            if not has:
                missing_by_class[cls] += 1

    return {
        "total_records": total,
        "skipped_missing_image": skipped,
        "missing_masks_by_class": missing_by_class,
    }


def preprocess_and_validate(
    project_root: str | Path = ".",
    output_dir: str | Path = "data/processed",
    seed: int = 42,
    run_summary_path: str | Path = "results/preprocess_run_summary.json",
) -> Dict[str, Any]:
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = Path(run_summary_path)

    paths = discover_datasets(project_root=project_root, raw_root="data/raw", strict=True)
    append_run_summary(
        run_summary_path,
        stage="dataset_discovery",
        status="ok",
        details={
            "aptos": str(paths.aptos),
            "idrid": str(paths.idrid),
            "ddr": str(paths.ddr),
        },
    )

    cls_df = build_classification_dataframe(paths)
    cls_split_paths = save_classification_splits(
        cls_df,
        output_dir=output_dir / "splits",
        seed=seed,
        folds=5,
    )
    append_run_summary(
        run_summary_path,
        stage="classification_splits",
        status="ok",
        details={
            "num_samples": int(len(cls_df)),
            "splits": {k: str(v) for k, v in cls_split_paths.items()},
        },
    )

    seg_records = load_segmentation_records(paths, run_summary_path=run_summary_path)
    seg_stats = validate_segmentation_pairing(seg_records)
    seg_split_paths = split_segmentation_records(seg_records, out_dir=output_dir / "splits", seed=seed)
    append_run_summary(
        run_summary_path,
        stage="segmentation_splits",
        status="ok",
        details={
            "num_records": len(seg_records),
            "pairing": seg_stats,
            "splits": {k: str(v) for k, v in seg_split_paths.items()},
        },
    )

    loc_df = load_localization_dataframe(paths.idrid)
    loc_split_paths = split_localization_dataframe(loc_df, out_dir=output_dir / "splits", seed=seed)
    append_run_summary(
        run_summary_path,
        stage="localization_splits",
        status="ok",
        details={
            "num_records": len(loc_df),
            "splits": {k: str(v) for k, v in loc_split_paths.items()},
        },
    )

    return {
        "dataset_paths": paths,
        "classification_samples": len(cls_df),
        "segmentation_records": len(seg_records),
        "localization_records": len(loc_df),
        "classification_split_paths": cls_split_paths,
        "segmentation_split_paths": seg_split_paths,
        "localization_split_paths": loc_split_paths,
    }
