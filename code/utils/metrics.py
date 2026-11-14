from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


EPS = 1e-7


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_prob = y_prob / row_sums
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    one_hot = label_binarize(y_true, classes=list(range(len(class_names))))
    roc_aucs = {}
    pr_aucs = {}
    for i, class_name in enumerate(class_names):
        positives = one_hot[:, i].sum()
        negatives = len(one_hot[:, i]) - positives
        if positives == 0 or negatives == 0:
            roc_aucs[f"roc_auc_{class_name}"] = float("nan")
            pr_aucs[f"pr_auc_{class_name}"] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(one_hot[:, i], y_prob[:, i])
        roc_aucs[f"roc_auc_{class_name}"] = float(auc(fpr, tpr))
        p, r, _ = precision_recall_curve(one_hot[:, i], y_prob[:, i])
        pr_aucs[f"pr_auc_{class_name}"] = float(auc(r, p))

    metrics.update(roc_aucs)
    metrics.update(pr_aucs)
    return metrics


def save_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    one_hot = label_binarize(y_true, classes=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        if one_hot[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(one_hot[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=class_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        if one_hot[:, i].sum() == 0:
            continue
        p, r, _ = precision_recall_curve(one_hot[:, i], y_prob[:, i])
        ax.plot(r, p, label=class_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=200)
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    path: str | Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    import pandas as pd

    df = pd.DataFrame(report).transpose()
    df.to_csv(path, index=True)


def dice_score(logits: np.ndarray, targets: np.ndarray) -> float:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.float32)
    targets = targets.astype(np.float32)
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return float((2 * inter + EPS) / (union + EPS))


def dice_torch(pred: "np.ndarray | torch.Tensor", target: "np.ndarray | torch.Tensor") -> float:
    import torch

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().float()
    else:
        pred = torch.tensor(pred).float()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().float()
    else:
        target = torch.tensor(target).float()
    pred = (pred > 0.5).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return float((2 * inter + EPS) / (union + EPS))


def iou_torch(pred: "np.ndarray | torch.Tensor", target: "np.ndarray | torch.Tensor") -> float:
    import torch

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().float()
    else:
        pred = torch.tensor(pred).float()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().float()
    else:
        target = torch.tensor(target).float()
    pred = (pred > 0.5).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return float((inter + EPS) / (union + EPS))


def segmentation_metrics(
    probs: "np.ndarray | torch.Tensor",
    targets: "np.ndarray | torch.Tensor",
    class_names: List[str],
) -> Dict[str, float]:
    import torch

    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
    else:
        probs = torch.tensor(probs)
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu()
    else:
        targets = torch.tensor(targets)

    probs = probs.float()
    targets = targets.float()
    result: Dict[str, float] = {}
    dices = []
    ious = []
    for c, class_name in enumerate(class_names):
        d = dice_torch(probs[:, c], targets[:, c])
        i = iou_torch(probs[:, c], targets[:, c])
        dices.append(d)
        ious.append(i)
        result[f"dice_{class_name}"] = d
        result[f"iou_{class_name}"] = i

    result["mean_dice"] = float(np.mean(dices))
    result["mean_iou"] = float(np.mean(ious))
    return result


def localization_metrics(
    pred_xy: np.ndarray,
    true_xy: np.ndarray,
    thresholds: Tuple[int, int] = (10, 20),
) -> Dict[str, float]:
    diff = pred_xy - true_xy
    dist = np.sqrt((diff ** 2).sum(axis=1))
    metrics = {"mean_pixel_error": float(np.mean(dist))}
    for th in thresholds:
        metrics[f"within_{th}px"] = float(np.mean(dist <= th))
    return metrics
