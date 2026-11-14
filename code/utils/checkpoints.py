from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


class CheckpointManager:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        best_metric: float,
        metadata: Dict[str, Any],
    ) -> Path:
        path = self.out_dir / f"{name}.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "metadata": metadata,
        }
        torch.save(payload, path)
        return path


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return payload
