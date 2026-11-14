from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class GPUInfo:
    name: str
    total_vram_gb: float
    cuda_version: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unavailable"


def get_gpu_info() -> Optional[GPUInfo]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return GPUInfo(
        name=props.name,
        total_vram_gb=props.total_memory / (1024 ** 3),
        cuda_version=torch.version.cuda or "unknown",
    )


def require_gpu(min_vram_gb: float = 8.0) -> GPUInfo:
    info = get_gpu_info()
    if info is None:
        raise RuntimeError(
            "GPU REQUIRED — aborting\n"
            "Minimum requirements: NVIDIA CUDA-compatible GPU with >=8GB VRAM (>=12GB recommended for high-resolution segmentation/EfficientNet-B5)."
        )
    if info.total_vram_gb + 0.05 < min_vram_gb:
        raise RuntimeError(
            "GPU REQUIRED — aborting\n"
            f"Detected: {info.name} with {info.total_vram_gb:.2f}GB VRAM.\n"
            f"Required minimum for this run: {min_vram_gb:.1f}GB VRAM.\n"
            "Baseline requirement: >=8GB VRAM (>=12GB recommended for larger backbones/high resolution)."
        )
    return info


def choose_batch_size(vram_gb: float, preferred: int, fallback_min: int = 1) -> int:
    if vram_gb >= 20:
        return max(preferred, 16)
    if vram_gb >= 12:
        return max(preferred, 12)
    if vram_gb >= 8:
        return max(fallback_min, min(preferred, 8))
    return fallback_min


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_run_summary(path: str | Path, stage: str, status: str, details: Dict[str, Any]) -> None:
    entry = {
        "timestamp_utc": utc_now_iso(),
        "stage": stage,
        "status": status,
        "details": details,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch
