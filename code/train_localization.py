from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.config import RunConfig, save_config
from code.dataset import LocalizationDataset, preprocess_and_validate
from code.models.localization import build_localization_model
from code.utils.checkpoints import CheckpointManager, load_checkpoint
from code.utils.experiment import OptionalWandB
from code.utils.logging_utils import CSVLogger, OptionalTensorBoard
from code.utils.schedulers import warmup_cosine_scheduler
from code.utils.system import (
    append_run_summary,
    get_git_commit,
    require_gpu,
    set_seed,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train optic-disc/fovea localization model.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--splits-dir", type=str, default="data/processed/splits")
    parser.add_argument("--results-dir", type=str, default="results/localization")
    parser.add_argument("--models-dir", type=str, default="models/localization")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--success-mean-error", type=float, default=40.0)
    parser.add_argument("--success-within20", type=float, default=0.80)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dr-localization")
    parser.add_argument("--run-summary", type=str, default="results/localization/run_summary.json")
    return parser.parse_args()


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = LocalizationDataset(train_df, image_size=image_size, train=True)
    val_ds = LocalizationDataset(val_df, image_size=image_size, train=False)
    test_ds = LocalizationDataset(test_df, image_size=image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _is_oom_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg) or ("cublas_status_alloc_failed" in msg)


def auto_detect_batch_size(
    image_size: int,
    device: torch.device,
    mixed_precision: bool = True,
    preferred: int = 16,
    max_probe: int = 48,
) -> int:
    # Probe descending sizes and pick the largest that fits a full forward+backward pass.
    candidates = [max_probe, 40, 32, 24, 20, preferred, 12, 10, 8, 6, 4, 2, 1]
    candidates = sorted({int(c) for c in candidates if int(c) > 0}, reverse=True)

    probe_model = build_localization_model(pretrained=False).to(device)
    probe_model.train()
    best = 1

    for bs in candidates:
        if bs > max_probe:
            continue
        x = None
        target = None
        pred = None
        loss = None
        try:
            torch.cuda.empty_cache()
            x = torch.randn(bs, 3, image_size, image_size, device=device)
            target = torch.rand(bs, 4, device=device)
            probe_model.zero_grad(set_to_none=True)
            with autocast(enabled=mixed_precision):
                pred = probe_model(x)
                loss = nn.SmoothL1Loss(beta=0.02)(pred[:, :4], target)
            loss.backward()
            best = bs
            break
        except RuntimeError as err:
            if not _is_oom_error(err):
                raise
        finally:
            if x is not None:
                del x
            if target is not None:
                del target
            if pred is not None:
                del pred
            if loss is not None:
                del loss
            probe_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    del probe_model
    torch.cuda.empty_cache()
    return max(1, best)


def make_gaussian_heatmaps(
    target: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
) -> torch.Tensor:
    b = target.shape[0]
    dtype = target.dtype
    device = target.device
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(ys, xs)
    yy = yy.view(1, 1, height, width)
    xx = xx.view(1, 1, height, width)

    kp = torch.stack(
        [
            target[:, [0, 2]] * float(max(width - 1, 1)),
            target[:, [1, 3]] * float(max(height - 1, 1)),
        ],
        dim=-1,
    )
    cx = kp[:, :, 0].view(b, 2, 1, 1)
    cy = kp[:, :, 1].view(b, 2, 1, 1)

    sigma2 = float(max(sigma, 1.0)) ** 2
    g = torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * sigma2))
    return g.clamp_(0.0, 1.0)


def localization_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    heatmaps: torch.Tensor | None = None,
) -> torch.Tensor:
    coords = pred[:, :4]
    coord_loss = nn.SmoothL1Loss(beta=0.01)(coords, target)

    pred_delta = torch.stack([coords[:, 0] - coords[:, 2], coords[:, 1] - coords[:, 3]], dim=1)
    true_delta = torch.stack([target[:, 0] - target[:, 2], target[:, 1] - target[:, 3]], dim=1)
    geom_loss = nn.SmoothL1Loss(beta=0.02)(pred_delta, true_delta)

    heatmap_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if heatmaps is not None:
        _, _, h, w = heatmaps.shape
        sigma = max(1.5, min(h, w) / 32.0)
        target_hm = make_gaussian_heatmaps(target=target, height=h, width=w, sigma=sigma)
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmaps, target_hm) + F.mse_loss(
            torch.sigmoid(heatmaps), target_hm
        )

    if pred.size(1) > 4:
        unc_reg = pred[:, 4:].mean()
        return coord_loss + 0.3 * geom_loss + 0.2 * heatmap_loss + 0.0005 * unc_reg
    return coord_loss + 0.3 * geom_loss + 0.2 * heatmap_loss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=mixed_precision):
            pred = model(images)
            heatmaps = getattr(model, "last_heatmaps", None)
            loss = localization_loss(pred, target, heatmaps=heatmaps)
        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        total += int(images.size(0))

    return {"loss": total_loss / max(total, 1)}


def compute_pixel_metrics(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    orig_sizes: np.ndarray,
) -> tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    pred_points = []
    true_points = []

    for i in range(pred_norm.shape[0]):
        w, h = float(orig_sizes[i, 0]), float(orig_sizes[i, 1])
        p = np.array(
            [
                [pred_norm[i, 0] * w, pred_norm[i, 1] * h],
                [pred_norm[i, 2] * w, pred_norm[i, 3] * h],
            ],
            dtype=np.float32,
        )
        t = np.array(
            [
                [true_norm[i, 0] * w, true_norm[i, 1] * h],
                [true_norm[i, 2] * w, true_norm[i, 3] * h],
            ],
            dtype=np.float32,
        )
        pred_points.append(p)
        true_points.append(t)

    pred_arr = np.concatenate(pred_points, axis=0) if pred_points else np.zeros((0, 2), dtype=np.float32)
    true_arr = np.concatenate(true_points, axis=0) if true_points else np.zeros((0, 2), dtype=np.float32)
    if len(pred_arr) == 0:
        metrics = {
            "mean_pixel_error": 0.0,
            "median_pixel_error": 0.0,
            "within_10px": 0.0,
            "within_20px": 0.0,
        }
        return metrics, pred_arr, true_arr, np.zeros((0,), dtype=np.float32)

    dist = np.sqrt(((pred_arr - true_arr) ** 2).sum(axis=1))
    metrics = {
        "mean_pixel_error": float(np.mean(dist)),
        "median_pixel_error": float(np.median(dist)),
        "within_10px": float(np.mean(dist <= 10.0)),
        "within_20px": float(np.mean(dist <= 20.0)),
    }
    return metrics, pred_arr, true_arr, dist.astype(np.float32)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0

    pred_norm_list: List[np.ndarray] = []
    true_norm_list: List[np.ndarray] = []
    orig_size_list: List[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        target_orig = batch.get("target_orig", batch["target"]).to(device, non_blocking=True)
        hflip_applied = batch.get("hflip_applied")
        if hflip_applied is not None:
            hflip_applied = hflip_applied.to(device, non_blocking=True)
        orig_size = batch["orig_size"].detach().cpu().numpy()

        with autocast(enabled=mixed_precision):
            pred = model(images)
            heatmaps = getattr(model, "last_heatmaps", None)
            loss = localization_loss(pred, target, heatmaps=heatmaps)

        pred_norm_t = pred[:, :4]
        if hflip_applied is not None:
            flip_mask = (hflip_applied > 0.5).float()
            pred_norm_t = pred_norm_t.clone()
            pred_norm_t[:, 0] = torch.where(flip_mask > 0, 1.0 - pred_norm_t[:, 0], pred_norm_t[:, 0])
            pred_norm_t[:, 2] = torch.where(flip_mask > 0, 1.0 - pred_norm_t[:, 2], pred_norm_t[:, 2])

        pred_norm = pred_norm_t.detach().cpu().numpy()
        pred_norm = np.clip(pred_norm, 0.0, 1.0)
        true_norm = target_orig.detach().cpu().numpy()

        total_loss += float(loss.item()) * images.size(0)
        total += int(images.size(0))

        pred_norm_list.append(pred_norm)
        true_norm_list.append(true_norm)
        orig_size_list.append(orig_size)

    if pred_norm_list:
        pred_norm_arr = np.concatenate(pred_norm_list, axis=0)
        true_norm_arr = np.concatenate(true_norm_list, axis=0)
        orig_size_arr = np.concatenate(orig_size_list, axis=0)
    else:
        pred_norm_arr = np.zeros((0, 4), dtype=np.float32)
        true_norm_arr = np.zeros((0, 4), dtype=np.float32)
        orig_size_arr = np.zeros((0, 2), dtype=np.float32)

    metrics, pred_xy, true_xy, dist = compute_pixel_metrics(pred_norm_arr, true_norm_arr, orig_size_arr)
    metrics["loss"] = total_loss / max(total, 1)
    return {
        "metrics": metrics,
        "pred_xy": pred_xy,
        "true_xy": true_xy,
        "distances": dist,
    }


def attempt_hparams(args: argparse.Namespace, attempt_idx: int) -> Dict[str, float]:
    if attempt_idx == 1:
        return {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
        }
    if attempt_idx == 2:
        return {
            "learning_rate": args.learning_rate * 0.5,
            "weight_decay": args.weight_decay * 0.5,
            "image_size": min(args.image_size + 64, 640),
        }
    return {
        "learning_rate": args.learning_rate * 1.5,
        "weight_decay": args.weight_decay * 0.1,
        "image_size": args.image_size,
    }


def run_attempt(
    attempt_idx: int,
    args: argparse.Namespace,
    model_used: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: Path,
    models_dir: Path,
    run_summary_path: Path,
    device: torch.device,
    gpu_info,
) -> Dict[str, Any]:
    hp = attempt_hparams(args, attempt_idx)
    attempt_name = f"attempt_{attempt_idx}"
    attempt_results_dir = results_dir / attempt_name
    attempt_models_dir = models_dir / attempt_name
    attempt_results_dir.mkdir(parents=True, exist_ok=True)
    attempt_models_dir.mkdir(parents=True, exist_ok=True)

    auto_batch = auto_detect_batch_size(
        image_size=int(hp["image_size"]),
        device=device,
        mixed_precision=True,
        preferred=16,
        max_probe=16,
    )
    batch_size = args.batch_size if args.batch_size > 0 else int(auto_batch)
    epochs = max(args.epochs, args.min_epochs)
    epochs = min(epochs, args.max_epochs)

    cfg = RunConfig(
        seed=args.seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        image_size=int(hp["image_size"]),
        num_workers=args.num_workers,
        mixed_precision=True,
        output_dir=str(attempt_results_dir),
        model_dir=str(attempt_models_dir),
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
    )
    save_config(attempt_results_dir / "train_config.yaml", cfg)

    train_loader, val_loader, test_loader = make_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = build_localization_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=cfg.warmup_epochs, total_epochs=cfg.epochs)
    scaler = GradScaler(enabled=cfg.mixed_precision)

    ckpt = CheckpointManager(attempt_models_dir)
    tb = OptionalTensorBoard(log_dir=attempt_results_dir / "tensorboard", enabled=cfg.use_tensorboard)
    wb = OptionalWandB(
        enabled=cfg.use_wandb,
        project=args.wandb_project,
        run_name=f"localization-{attempt_name}",
        config=cfg.to_dict(),
    )
    csv_logger = CSVLogger(attempt_results_dir / "epoch_metrics.csv")

    best_metric = float("inf")
    best_epoch = -1
    wait = 0

    append_run_summary(
        run_summary_path,
        stage="training_start",
        status="ok",
        details={
            "attempt": attempt_idx,
            "model_used": model_used,
            "num_train": len(train_df),
            "num_val": len(val_df),
            "num_test": len(test_df),
            "batch_size": cfg.batch_size,
            "image_size": cfg.image_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "gpu": gpu_info.name,
            "gpu_vram_gb": round(gpu_info.total_vram_gb, 2),
        },
    )

    for epoch in range(1, cfg.epochs + 1):
        try:
            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                mixed_precision=cfg.mixed_precision,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cfg.batch_size > 1:
                torch.cuda.empty_cache()
                cfg.batch_size = max(1, cfg.batch_size // 2)
                train_loader, val_loader, test_loader = make_loaders(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    image_size=cfg.image_size,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                )
                append_run_summary(
                    run_summary_path,
                    stage="oom_recovery",
                    status="ok",
                    details={"attempt": attempt_idx, "epoch": epoch, "new_batch_size": cfg.batch_size},
                )
                train_stats = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    mixed_precision=cfg.mixed_precision,
                )
            else:
                raise

        val_eval = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            mixed_precision=cfg.mixed_precision,
        )
        val_metrics = val_eval["metrics"]
        metric = float(val_metrics["mean_pixel_error"])

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": float(val_metrics.get("loss", 0.0)),
            "val_mean_pixel_error": float(val_metrics["mean_pixel_error"]),
            "val_median_pixel_error": float(val_metrics["median_pixel_error"]),
            "val_within_10px": float(val_metrics["within_10px"]),
            "val_within_20px": float(val_metrics["within_20px"]),
            "lr": optimizer.param_groups[0]["lr"],
        }
        csv_logger.log(row)
        tb.add_scalar("train/loss", row["train_loss"], epoch)
        tb.add_scalar("val/loss", row["val_loss"], epoch)
        tb.add_scalar("val/mean_pixel_error", row["val_mean_pixel_error"], epoch)
        tb.add_scalar("val/median_pixel_error", row["val_median_pixel_error"], epoch)
        tb.add_scalar("val/within_10px", row["val_within_10px"], epoch)
        tb.add_scalar("val/within_20px", row["val_within_20px"], epoch)
        tb.add_scalar("lr", row["lr"], epoch)
        wb.log(row, step=epoch)

        metadata = {
            "task": "localization",
            "model_name": "localization_direct_regression_v3",
            "model_used": model_used,
            "epoch": epoch,
            "date": utc_now_iso(),
            "git_commit": get_git_commit(),
            "seed": cfg.seed,
            "image_size": cfg.image_size,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
        }

        ckpt.save(
            name=f"epoch_{epoch:03d}",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=-metric,
            metadata=metadata,
        )

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            wait = 0
            ckpt.save(
                name="best",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=-metric,
                metadata=metadata,
            )
        else:
            wait += 1

        scheduler.step()
        if epoch >= args.min_epochs and wait >= cfg.patience:
            break

    ckpt.save(
        name="last",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=best_epoch if best_epoch > 0 else cfg.epochs,
        best_metric=-best_metric,
        metadata={
            "task": "localization",
            "model_name": "localization_direct_regression_v3",
            "model_used": model_used,
            "date": utc_now_iso(),
            "git_commit": get_git_commit(),
            "seed": cfg.seed,
        },
    )

    load_checkpoint(attempt_models_dir / "best.pt", model, device)
    final_val_eval = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        mixed_precision=cfg.mixed_precision,
    )
    final_test_eval = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        mixed_precision=cfg.mixed_precision,
    )
    final_val_metrics = final_val_eval["metrics"]
    final_test_metrics = final_test_eval["metrics"]

    write_json(attempt_results_dir / "val_metrics.json", final_val_metrics)
    write_json(attempt_results_dir / "test_metrics.json", final_test_metrics)

    append_run_summary(
        run_summary_path,
        stage="training_finish",
        status="ok",
        details={
            "attempt": attempt_idx,
            "best_epoch": best_epoch,
            "best_val_mean_pixel_error": best_metric,
            "best_val_within_20px": float(final_val_metrics["within_20px"]),
            "final_val_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
            "best_checkpoint": str(attempt_models_dir / "best.pt"),
        },
    )

    tb.close()
    wb.close()

    return {
        "attempt": attempt_idx,
        "attempt_name": attempt_name,
        "best_epoch": best_epoch,
        "best_checkpoint": str(attempt_models_dir / "best.pt"),
        "model_dir": str(attempt_models_dir),
        "result_dir": str(attempt_results_dir),
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "batch_size": cfg.batch_size,
        "image_size": cfg.image_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
    }


def promote_attempt_outputs(best_attempt: Dict[str, Any], results_dir: Path, models_dir: Path) -> None:
    src_res = Path(best_attempt["result_dir"])
    src_mod = Path(best_attempt["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for name in ["epoch_metrics.csv", "train_config.yaml", "val_metrics.json", "test_metrics.json"]:
        src = src_res / name
        if src.exists():
            shutil.copy2(src, results_dir / name)

    for name in ["best.pt", "last.pt"]:
        src = src_mod / name
        if src.exists():
            shutil.copy2(src, models_dir / name)


@torch.no_grad()
def save_visual_checks(
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    image_size: int,
    out_dir: Path,
    device: torch.device,
    max_samples: int = 20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = val_df.head(max_samples).reset_index(drop=True)
    ds = LocalizationDataset(subset, image_size=image_size, train=False)

    for i in range(len(ds)):
        sample = ds[i]
        x = sample["image"].unsqueeze(0).to(device)
        pred = model(x)[0, :4].detach().cpu().numpy()
        if float(sample.get("hflip_applied", torch.tensor(0.0)).item()) > 0.5:
            pred[0] = 1.0 - pred[0]
            pred[2] = 1.0 - pred[2]
        pred = np.clip(pred, 0.0, 1.0)

        row = subset.iloc[i]
        orig = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if orig is None:
            continue
        h, w = orig.shape[:2]

        gt_od = (float(row["od_x"]), float(row["od_y"]))
        gt_fv = (float(row["fv_x"]), float(row["fv_y"]))
        pr_od = (float(pred[0] * w), float(pred[1] * h))
        pr_fv = (float(pred[2] * w), float(pred[3] * h))

        od_err = float(np.hypot(pr_od[0] - gt_od[0], pr_od[1] - gt_od[1]))
        fv_err = float(np.hypot(pr_fv[0] - gt_fv[0], pr_fv[1] - gt_fv[1]))
        mean_err = (od_err + fv_err) / 2.0

        vis = orig.copy()
        cv2.circle(vis, (int(gt_od[0]), int(gt_od[1])), 8, (0, 255, 0), 2)
        cv2.circle(vis, (int(gt_fv[0]), int(gt_fv[1])), 8, (255, 0, 0), 2)
        cv2.drawMarker(vis, (int(pr_od[0]), int(pr_od[1])), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.drawMarker(vis, (int(pr_fv[0]), int(pr_fv[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.line(vis, (int(gt_od[0]), int(gt_od[1])), (int(pr_od[0]), int(pr_od[1])), (0, 200, 255), 1)
        cv2.line(vis, (int(gt_fv[0]), int(gt_fv[1])), (int(pr_fv[0]), int(pr_fv[1])), (0, 0, 255), 1)

        text1 = f"OD err: {od_err:.2f}px | FV err: {fv_err:.2f}px | Mean: {mean_err:.2f}px"
        text2 = "GT: OD(green), FV(blue) | Pred: OD(yellow X), FV(red X)"
        cv2.rectangle(vis, (12, 12), (min(vis.shape[1] - 12, 1100), 74), (0, 0, 0), -1)
        cv2.putText(vis, text1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text2, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        out_path = out_dir / f"{i:02d}_{row['image_id']}.png"
        cv2.imwrite(str(out_path), vis)


def meets_success(metrics: Dict[str, float], max_mean_error: float, min_within20: float) -> bool:
    return (metrics["mean_pixel_error"] < max_mean_error) and (metrics["within_20px"] >= min_within20)


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root)
    splits_dir = project_root / args.splits_dir
    results_dir = project_root / args.results_dir
    models_dir = project_root / args.models_dir
    run_summary_path = project_root / args.run_summary

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = require_gpu(min_vram_gb=8.0)
    device = torch.device("cuda")
    set_seed(args.seed, deterministic=True)

    if not (splits_dir / "localization_train.csv").exists():
        preprocess_and_validate(project_root=project_root, output_dir=project_root / "data/processed", seed=args.seed)

    train_df = pd.read_csv(splits_dir / "localization_train.csv")
    val_df = pd.read_csv(splits_dir / "localization_val.csv")
    test_df = pd.read_csv(splits_dir / "localization_test.csv")
    if len(train_df) == 0:
        raise RuntimeError("No localization training records found after preprocessing.")

    total_samples = len(train_df) + len(val_df) + len(test_df)
    model_used = "heatmap_regression" if total_samples > 500 else "direct_regression"

    attempts: List[Dict[str, Any]] = []
    success_attempt: Dict[str, Any] | None = None
    max_attempts = max(1, min(3, int(args.max_attempts)))

    for attempt_idx in range(1, max_attempts + 1):
        result = run_attempt(
            attempt_idx=attempt_idx,
            args=args,
            model_used=model_used,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            results_dir=results_dir,
            models_dir=models_dir,
            run_summary_path=run_summary_path,
            device=device,
            gpu_info=gpu_info,
        )
        attempts.append(result)
        if meets_success(
            result["val_metrics"],
            max_mean_error=args.success_mean_error,
            min_within20=args.success_within20,
        ):
            success_attempt = result
            break

    best_attempt = success_attempt
    if best_attempt is None:
        best_attempt = sorted(attempts, key=lambda r: r["val_metrics"]["mean_pixel_error"])[0]

    promote_attempt_outputs(best_attempt, results_dir=results_dir, models_dir=models_dir)

    best_model = build_localization_model(pretrained=False).to(device)
    load_checkpoint(models_dir / "best.pt", best_model, device)
    best_model.eval()
    save_visual_checks(
        model=best_model,
        val_df=val_df,
        image_size=int(best_attempt["image_size"]),
        out_dir=results_dir / "visual_checks",
        device=device,
        max_samples=20,
    )

    final_val_metrics = best_attempt["val_metrics"]
    final_test_metrics = best_attempt["test_metrics"]

    model_meta = {
        "model_name": "dr_localizer",
        "encoder": "resnet18",
        "model_used": model_used,
        "epochs": max(args.epochs, args.min_epochs),
        "best_metric": float(final_val_metrics["mean_pixel_error"]),
        "git_commit": get_git_commit(),
        "date": utc_now_iso(),
        "seed": args.seed,
        "dataset_versions": {"idrid": "local"},
    }
    write_json(models_dir / "metadata.json", model_meta)

    root_meta_path = project_root / "models/metadata.json"
    root_meta = {}
    if root_meta_path.exists():
        try:
            root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
        except Exception:
            root_meta = {}
    root_meta["localization"] = model_meta
    write_json(root_meta_path, root_meta)

    fixes_applied = [
        "Applied keypoint-aware localization augmentations so coordinates transform with images.",
        "Replaced localization head with a spatially-aware soft-argmax direct-regression decoder (OD/FV heatmaps -> coordinates).",
        "Switched localization objective to SmoothL1 coordinate regression with geometric consistency and heatmap supervision.",
        "Standardized validation/test metrics strictly in pixel space (mean/median/within10/within20).",
        "Added automatic GPU batch-size probing (largest fitting forward/backward batch).",
        "Enabled up to 3 automatic retraining attempts with adjusted hyperparameters.",
        "Generated 20-sample visual verification overlays for predicted vs ground-truth keypoints.",
    ]

    status = "SUCCESS" if meets_success(final_val_metrics, args.success_mean_error, args.success_within20) else "FAILED"
    final_report = {
        "model_used": model_used,
        "epochs_trained": int(max(args.epochs, args.min_epochs)),
        "mean_pixel_error": float(final_val_metrics["mean_pixel_error"]),
        "median_pixel_error": float(final_val_metrics["median_pixel_error"]),
        "within10px": float(final_val_metrics["within_10px"]),
        "within20px": float(final_val_metrics["within_20px"]),
        "fixes_applied": fixes_applied,
        "status": status,
        "attempts_ran": len(attempts),
        "selected_attempt": int(best_attempt["attempt"]),
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
    }
    write_json(results_dir / "final_report.json", final_report)

    append_run_summary(
        run_summary_path,
        stage="run_complete",
        status="ok" if status == "SUCCESS" else "failed",
        details={
            "model_used": model_used,
            "attempts_ran": len(attempts),
            "selected_attempt": int(best_attempt["attempt"]),
            "final_val_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
            "status": status,
        },
    )

    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
