from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RunConfig:
    seed: int = 42
    epochs: int = 120
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 5
    patience: int = 20
    image_size: int = 512
    num_workers: int = 4
    mixed_precision: bool = True
    grad_accum_steps: int = 1
    output_dir: str = "results"
    model_dir: str = "models"
    folds: int = 5
    fold_index: Optional[int] = None
    use_tensorboard: bool = False
    use_wandb: bool = False
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(payload.pop("extra"))
        return payload


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
LESION_CLASSES = [
    "microaneurysms",
    "hemorrhages",
    "hard_exudates",
    "soft_exudates",
    "optic_disc",
]


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f) or {}
        elif path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    known = {k: v for k, v in data.items() if k in RunConfig.__dataclass_fields__}
    extra = {k: v for k, v in data.items() if k not in known}
    cfg = RunConfig(**known)
    cfg.extra = extra
    return cfg


def save_config(path: str | Path, cfg: RunConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    elif path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")
