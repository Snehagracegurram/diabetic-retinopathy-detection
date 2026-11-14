from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.config import LESION_CLASSES, RunConfig, save_config
from code.dataset import SegmentationDataset, preprocess_and_validate
from code.models.segmentation import build_segmentation_model
from code.utils.checkpoints import CheckpointManager, load_checkpoint
from code.utils.experiment import OptionalWandB
from code.utils.logging_utils import CSVLogger, OptionalTensorBoard
from code.utils.losses import DiceBCELoss
from code.utils.metrics import segmentation_metrics
from code.utils.schedulers import warmup_cosine_scheduler
from code.utils.system import (
    append_run_summary,
    choose_batch_size,
    get_git_commit,
    require_gpu,
    set_seed,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DR lesion segmentation model (U-Net).")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--splits-dir", type=str, default="data/processed/splits")
    parser.add_argument("--results-dir", type=str, default="results/segmentation")
    parser.add_argument("--models-dir", type=str, default="models/segmentation")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dr-segmentation")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt) to resume from.")
    parser.add_argument("--run-summary", type=str, default="results/segmentation/run_summary.json")
    return parser.parse_args()


def load_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_loaders(
    train_records: List[dict],
    val_records: List[dict],
    test_records: List[dict],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SegmentationDataset(train_records, image_size=image_size, train=True)
    val_ds = SegmentationDataset(val_records, image_size=image_size, train=False)
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
    test_ds = SegmentationDataset(test_records, image_size=image_size, train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), 1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=mixed_precision):
            logits = model(images)
            loss = criterion(logits, masks) / grad_accum_steps

        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_val = float(loss.item() * grad_accum_steps)
        total_loss += loss_val * images.size(0)
        total += int(images.size(0))

    return {"loss": total_loss / max(total, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    probs_list = []
    target_list = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=mixed_precision):
            logits = model(images)
            loss = criterion(logits, masks)

        probs = torch.sigmoid(logits)

        total_loss += float(loss.item()) * images.size(0)
        total += int(images.size(0))

        probs_list.append(probs.detach().cpu())
        target_list.append(masks.detach().cpu())

    if probs_list:
        probs_tensor = torch.cat(probs_list, dim=0)
        target_tensor = torch.cat(target_list, dim=0)
        metrics = segmentation_metrics(probs_tensor, target_tensor, LESION_CLASSES)
    else:
        metrics = {"mean_dice": 0.0, "mean_iou": 0.0}

    metrics["loss"] = total_loss / max(total, 1)
    return metrics


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

    epochs = max(args.epochs, args.min_epochs)
    epochs = min(epochs, args.max_epochs)

    set_seed(args.seed, deterministic=True)

    if not (splits_dir / "segmentation_train.json").exists():
        preprocess_and_validate(project_root=project_root, output_dir=project_root / "data/processed", seed=args.seed)

    train_records = load_records(splits_dir / "segmentation_train.json")
    val_records = load_records(splits_dir / "segmentation_val.json")
    test_records = load_records(splits_dir / "segmentation_test.json")

    if len(train_records) == 0:
        raise RuntimeError("No segmentation training records found after preprocessing.")

    auto_batch = choose_batch_size(gpu_info.total_vram_gb, preferred=6)
    batch_size = args.batch_size if args.batch_size > 0 else auto_batch

    cfg = RunConfig(
        seed=args.seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        image_size=args.image_size,
        num_workers=args.num_workers,
        mixed_precision=True,
        grad_accum_steps=max(1, args.grad_accum_steps),
        output_dir=str(results_dir),
        model_dir=str(models_dir),
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
    )
    save_config(results_dir / "train_config.yaml", cfg)

    train_loader, val_loader, test_loader = make_loaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = build_segmentation_model(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        classes=len(LESION_CLASSES),
    ).to(device)

    criterion = DiceBCELoss(smooth=1.0, bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=cfg.warmup_epochs, total_epochs=cfg.epochs)
    scaler = GradScaler(enabled=cfg.mixed_precision)

    ckpt = CheckpointManager(models_dir)
    tb = OptionalTensorBoard(log_dir=results_dir / "tensorboard", enabled=cfg.use_tensorboard)
    wb = OptionalWandB(
        enabled=cfg.use_wandb,
        project=args.wandb_project,
        run_name="segmentation",
        config=cfg.to_dict(),
    )
    csv_logger = CSVLogger(results_dir / "epoch_metrics.csv")

    best_metric = -1.0
    best_epoch = -1
    wait = 0
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        best_metric = float(payload.get("best_metric", best_metric))
        best_epoch = int(payload.get("epoch", 0))
        start_epoch = best_epoch + 1
        append_run_summary(
            run_summary_path,
            stage="resume",
            status="ok",
            details={"resume_path": str(resume_path), "start_epoch": start_epoch, "best_metric": best_metric},
        )

    append_run_summary(
        run_summary_path,
        stage="training_start",
        status="ok",
        details={
            "num_train": len(train_records),
            "num_val": len(val_records),
            "num_test": len(test_records),
            "batch_size": cfg.batch_size,
            "gpu": gpu_info.name,
            "gpu_vram_gb": round(gpu_info.total_vram_gb, 2),
        },
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        try:
            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                grad_accum_steps=cfg.grad_accum_steps,
                mixed_precision=cfg.mixed_precision,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cfg.batch_size > 1:
                torch.cuda.empty_cache()
                cfg.batch_size = max(1, cfg.batch_size // 2)
                train_loader, val_loader, test_loader = make_loaders(
                    train_records=train_records,
                    val_records=val_records,
                    test_records=test_records,
                    image_size=cfg.image_size,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                )
                append_run_summary(
                    run_summary_path,
                    stage="oom_recovery",
                    status="ok",
                    details={"epoch": epoch, "new_batch_size": cfg.batch_size},
                )
                train_stats = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    scaler=scaler,
                    device=device,
                    grad_accum_steps=cfg.grad_accum_steps,
                    mixed_precision=cfg.mixed_precision,
                )
            else:
                raise

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            mixed_precision=cfg.mixed_precision,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": float(val_metrics.get("loss", 0.0)),
            "val_mean_dice": float(val_metrics.get("mean_dice", 0.0)),
            "val_mean_iou": float(val_metrics.get("mean_iou", 0.0)),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for cls in LESION_CLASSES:
            row[f"val_dice_{cls}"] = float(val_metrics.get(f"dice_{cls}", 0.0))
            row[f"val_iou_{cls}"] = float(val_metrics.get(f"iou_{cls}", 0.0))

        csv_logger.log(row)
        tb.add_scalar("train/loss", row["train_loss"], epoch)
        tb.add_scalar("val/loss", row["val_loss"], epoch)
        tb.add_scalar("val/mean_dice", row["val_mean_dice"], epoch)
        tb.add_scalar("val/mean_iou", row["val_mean_iou"], epoch)
        tb.add_scalar("lr", row["lr"], epoch)
        wb.log(row, step=epoch)

        metadata = {
            "task": "segmentation",
            "encoder": "efficientnet-b3",
            "num_classes": len(LESION_CLASSES),
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
            best_metric=best_metric,
            metadata=metadata,
        )

        metric = float(val_metrics.get("mean_dice", 0.0))
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            wait = 0
            ckpt.save(
                name="best",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
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
        best_metric=best_metric,
        metadata={
            "task": "segmentation",
            "encoder": "efficientnet-b3",
            "num_classes": len(LESION_CLASSES),
            "date": utc_now_iso(),
            "git_commit": get_git_commit(),
            "seed": cfg.seed,
        },
    )

    load_checkpoint(models_dir / "best.pt", model, device)
    final_val_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        mixed_precision=cfg.mixed_precision,
    )
    final_test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        mixed_precision=cfg.mixed_precision,
    )

    write_json(results_dir / "val_metrics.json", final_val_metrics)
    write_json(results_dir / "test_metrics.json", final_test_metrics)

    per_class = pd.DataFrame(
        {
            "class": LESION_CLASSES,
            "dice": [float(final_val_metrics.get(f"dice_{c}", 0.0)) for c in LESION_CLASSES],
            "iou": [float(final_val_metrics.get(f"iou_{c}", 0.0)) for c in LESION_CLASSES],
        }
    )
    per_class.to_csv(results_dir / "per_class_segmentation_metrics.csv", index=False)

    model_meta = {
        "model_name": "dr_segmenter",
        "encoder": "efficientnet-b3",
        "epochs": epochs,
        "best_metric": best_metric,
        "git_commit": get_git_commit(),
        "date": utc_now_iso(),
        "seed": args.seed,
        "dataset_versions": {
            "idrid": "local",
            "ddr": "local",
        },
    }
    write_json(models_dir / "metadata.json", model_meta)
    root_meta_path = project_root / "models/metadata.json"
    root_meta = {}
    if root_meta_path.exists():
        try:
            root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
        except Exception:
            root_meta = {}
    root_meta["segmentation"] = model_meta
    write_json(root_meta_path, root_meta)

    append_run_summary(
        run_summary_path,
        stage="training_finish",
        status="ok",
        details={
            "best_epoch": best_epoch,
            "best_val_mean_dice": best_metric,
            "final_val_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
        },
    )

    tb.close()
    wb.close()

    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_mean_dice": best_metric,
                "results_dir": str(results_dir),
                "models_dir": str(models_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
