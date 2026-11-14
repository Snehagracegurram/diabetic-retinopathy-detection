from __future__ import annotations

from typing import Dict, Optional


class OptionalWandB:
    def __init__(self, enabled: bool, project: str, run_name: str, config: Dict):
        self.enabled = enabled
        self._run = None
        if enabled:
            try:
                import wandb

                self._run = wandb.init(project=project, name=run_name, config=config)
            except Exception:
                self.enabled = False
                self._run = None

    def log(self, data: Dict, step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(data, step=step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
