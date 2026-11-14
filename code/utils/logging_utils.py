from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0
        self._fieldnames: Optional[list[str]] = None

    def log(self, row: Dict[str, float]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


class OptionalTensorBoard:
    def __init__(self, log_dir: str | Path, enabled: bool):
        self.enabled = enabled
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(log_dir))
            except Exception:
                self.writer = None
                self.enabled = False

    def add_scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
