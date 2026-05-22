from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from core.scoring import DEFAULT_SCORE_FUSION_WEIGHTS


@dataclass
class BinaryMetrics:
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float
    false_positive_rate: float
    false_negative_rate: float
    best_threshold: float
    product_score: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(labels, scores >= threshold, zero_division=0) for threshold in thresholds]
    index = int(np.argmax(f1s))
    return float(thresholds[index]), float(f1s[index])


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def compute_binary_metrics(labels: list[float], scores: list[float]) -> BinaryMetrics:
    y_true = np.asarray(labels).astype(int)
    y_score = np.asarray(scores).astype(float)
    threshold, best_f1 = threshold_sweep(y_true, y_score)
    y_pred = y_score >= threshold
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    eer = equal_error_rate(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    false_positive_rate = float(((y_pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum()))
    false_negative_rate = float(((y_pred == 0) & (y_true == 1)).sum() / max(1, (y_true == 1).sum()))
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    return BinaryMetrics(
        auc=float(auc),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=best_f1,
        eer=eer,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        best_threshold=threshold,
        product_score=float(0.55 * auc + 0.35 * best_f1 + 0.10 * (1.0 - false_positive_rate)),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def fuse_detection_scores(
    real_fake: np.ndarray,
    inswapper: np.ndarray,
    boundary: np.ndarray,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    weights = weights or DEFAULT_SCORE_FUSION_WEIGHTS
    return (
        weights["real_fake"] * real_fake
        + weights["inswapper"] * inswapper
        + weights["boundary"] * boundary
    )
