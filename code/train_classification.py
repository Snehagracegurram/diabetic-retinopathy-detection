from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.config import CLASS_NAMES, RunConfig, save_config
from code.dataset import ClassificationDataset, preprocess_and_validate
from code.models.classifier import build_classifier
from code.utils.checkpoints import CheckpointManager, load_checkpoint
from code.utils.experiment import OptionalWandB
from code.utils.logging_utils import CSVLogger, OptionalTensorBoard
from code.utils.metrics import (
    classification_metrics,
    save_classification_plots,
    save_classification_report,
)
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
    parser = argparse.ArgumentParser(description="Train DR classification model (EfficientNet).")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--splits-dir", type=str, default="data/processed/splits")
    parser.add_argument("--results-dir", type=str, default="results/classification")
    parser.add_argument("--models-dir", type=str, default="models/classification")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3", choices=["efficientnet_b3", "efficientnet_b5"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto-detect from GPU VRAM.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--use-kfold", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dr-classification")
    parser.add_argument("--run-summary", type=str, default="results/classification/run_summary.json")
    return parser.parse_args()


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ClassificationDataset(train_df, image_size=image_size, train=True)
    val_ds = ClassificationDataset(val_df, image_size=image_size, train=False)
    test_ds = ClassificationDataset(test_df, image_size=image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    optimizer.zero_grad(set_to_none=True)

    skipped_batches = 0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), 1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=mixed_precision):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum_steps
        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.item() * grad_accum_steps)
        total_loss += loss_value * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "skipped_batches": skipped_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    total_loss = 0.0
    total = 0

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=mixed_precision):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            continue

        probs = torch.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        preds = torch.argmax(probs, dim=1)

        total_loss += float(loss.item()) * labels.size(0)
        total += int(labels.size(0))

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(CLASS_NAMES)))

    metrics = classification_metrics(y_true, y_pred, y_prob, CLASS_NAMES) if len(y_true) > 0 else {}
    metrics["loss"] = total_loss / max(total, 1)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }


def _load_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    return pd.read_csv(path)


def train_fold(
    fold_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: RunConfig,
    args: argparse.Namespace,
    device: torch.device,
    results_dir: Path,
    models_dir: Path,
    run_summary_path: Path,
    gpu_info,
) -> Dict[str, float]:
    fold_result_dir = results_dir / fold_name
    fold_model_dir = models_dir / fold_name
    fold_result_dir.mkdir(parents=True, exist_ok=True)
    fold_model_dir.mkdir(parents=True, exist_ok=True)

    tb = OptionalTensorBoard(log_dir=fold_result_dir / "tensorboard", enabled=cfg.use_tensorboard)
    wb = OptionalWandB(
        enabled=cfg.use_wandb,
        project=args.wandb_project,
        run_name=f"classification-{fold_name}",
        config=cfg.to_dict(),
    )
    csv_logger = CSVLogger(fold_result_dir / "epoch_metrics.csv")

    train_loader, val_loader, test_loader = make_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = build_classifier(
        backbone=args.backbone,
        num_classes=len(CLASS_NAMES),
        dropout=0.4,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
    )
    scaler = GradScaler(enabled=cfg.mixed_precision)

    ckpt = CheckpointManager(fold_model_dir)
    best_metric = -1.0
    best_epoch = -1
    wait = 0

    append_run_summary(
        run_summary_path,
        stage="training_start",
        status="ok",
        details={
            "fold": fold_name,
            "num_train": len(train_df),
            "num_val": len(val_df),
            "num_test": len(test_df),
            "backbone": args.backbone,
            "batch_size": cfg.batch_size,
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
                    details={"fold": fold_name, "new_batch_size": cfg.batch_size, "epoch": epoch},
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

        val_eval = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            mixed_precision=cfg.mixed_precision,
        )
        val_metrics = val_eval["metrics"]
        macro_f1 = float(val_metrics.get("macro_f1", 0.0))

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "train_skipped_batches": train_stats.get("skipped_batches", 0),
            "val_loss": float(val_metrics.get("loss", 0.0)),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_macro_f1": macro_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        csv_logger.log(row)
        tb.add_scalar("train/loss", row["train_loss"], epoch)
        tb.add_scalar("train/acc", row["train_acc"], epoch)
        tb.add_scalar("val/loss", row["val_loss"], epoch)
        tb.add_scalar("val/acc", row["val_accuracy"], epoch)
        tb.add_scalar("val/macro_f1", row["val_macro_f1"], epoch)
        tb.add_scalar("lr", row["lr"], epoch)
        wb.log(row, step=epoch)

        metadata = {
            "task": "classification",
            "fold": fold_name,
            "backbone": args.backbone,
            "num_classes": len(CLASS_NAMES),
            "epoch": epoch,
            "date": utc_now_iso(),
            "git_commit": get_git_commit(),
            "seed": cfg.seed,
            "image_size": cfg.image_size,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
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

        if macro_f1 > best_metric:
            best_metric = macro_f1
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
            "task": "classification",
            "fold": fold_name,
            "backbone": args.backbone,
            "num_classes": len(CLASS_NAMES),
            "date": utc_now_iso(),
            "git_commit": get_git_commit(),
            "seed": cfg.seed,
        },
    )

    best_path = fold_model_dir / "best.pt"
    payload = load_checkpoint(best_path, model, device)

    test_eval = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        mixed_precision=cfg.mixed_precision,
    )

    y_true = test_eval["y_true"]
    y_pred = test_eval["y_pred"]
    y_prob = test_eval["y_prob"]

    save_classification_report(y_true, y_pred, CLASS_NAMES, fold_result_dir / "classification_report.csv")
    save_classification_plots(y_true, y_pred, y_prob, CLASS_NAMES, fold_result_dir)

    test_metrics = dict(test_eval["metrics"])
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_macro_f1"] = best_metric
    write_json(fold_result_dir / "test_metrics.json", test_metrics)

    append_run_summary(
        run_summary_path,
        stage="training_finish",
        status="ok",
        details={
            "fold": fold_name,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_metric,
            "test_accuracy": float(test_metrics.get("accuracy", 0.0)),
            "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
            "best_checkpoint": str(best_path),
        },
    )

    tb.close()
    wb.close()
    return test_metrics


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    results_dir = project_root / args.results_dir
    models_dir = project_root / args.models_dir
    splits_dir = project_root / args.splits_dir
    run_summary_path = project_root / args.run_summary

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = require_gpu(min_vram_gb=8.0)
    device = torch.device("cuda")

    epochs = max(args.epochs, args.min_epochs)
    epochs = min(epochs, args.max_epochs)

    set_seed(args.seed, deterministic=True)

    # Ensure split files exist.
    if not (splits_dir / "classification_train.csv").exists():
        preprocess_and_validate(project_root=project_root, output_dir=project_root / "data/processed", seed=args.seed)

    auto_batch = choose_batch_size(gpu_info.total_vram_gb, preferred=8 if args.backbone == "efficientnet_b3" else 4)
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
        folds=args.folds,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
    )

    save_config(results_dir / "train_config.yaml", cfg)

    if args.use_kfold:
        fold_ids = list(range(args.folds))
    else:
        fold_ids = [None]

    test_df = _load_split_csv(splits_dir / "classification_test.csv")

    all_metrics = []
    for fold_id in fold_ids:
        if fold_id is None:
            fold_name = "single_split"
            train_df = _load_split_csv(splits_dir / "classification_train.csv")
            val_df = _load_split_csv(splits_dir / "classification_val.csv")
        else:
            fold_name = f"fold_{fold_id}"
            train_df = _load_split_csv(splits_dir / f"classification_fold{fold_id}_train.csv")
            val_df = _load_split_csv(splits_dir / f"classification_fold{fold_id}_val.csv")

        metrics = train_fold(
            fold_name=fold_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cfg=cfg,
            args=args,
            device=device,
            results_dir=results_dir,
            models_dir=models_dir,
            run_summary_path=run_summary_path,
            gpu_info=gpu_info,
        )
        metrics["fold"] = fold_name
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)

    aggregate = {
        "accuracy_mean": float(metrics_df["accuracy"].mean()) if "accuracy" in metrics_df else 0.0,
        "accuracy_std": float(metrics_df["accuracy"].std()) if "accuracy" in metrics_df else 0.0,
        "macro_f1_mean": float(metrics_df["macro_f1"].mean()) if "macro_f1" in metrics_df else 0.0,
        "macro_f1_std": float(metrics_df["macro_f1"].std()) if "macro_f1" in metrics_df else 0.0,
    }

    write_json(results_dir / "aggregate_metrics.json", aggregate)

    model_meta = {
        "model_name": "dr_classifier",
        "encoder": args.backbone,
        "epochs": epochs,
        "best_metric": aggregate["macro_f1_mean"],
        "git_commit": get_git_commit(),
        "date": utc_now_iso(),
        "seed": args.seed,
        "dataset_versions": {
            "aptos": "local",
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
    root_meta["classification"] = model_meta
    write_json(root_meta_path, root_meta)

    append_run_summary(
        run_summary_path,
        stage="run_complete",
        status="ok",
        details={"aggregate": aggregate, "metadata": model_meta},
    )

    print(json.dumps({"aggregate": aggregate, "models_dir": str(models_dir), "results_dir": str(results_dir)}, indent=2))


if __name__ == "__main__":
    main()
