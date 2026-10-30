"""Các hàm đánh giá và so sánh mô hình phân loại nhị phân."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Tính các thước đo từ xác suất dự đoán của lớp 1.

    Returns:
        Từ điển gồm Accuracy, Precision, Recall, F1, ROC-AUC, log loss
        và bốn ô của ma trận nhầm lẫn (TN, FP, FN, TP).
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def majority_baseline(y_true: np.ndarray) -> float:
    """Accuracy của mô hình luôn dự đoán lớp đông hơn (mốc tối thiểu)."""
    return float(max(np.mean(y_true), 1 - np.mean(y_true)))


def time_fit(fit_fn: Callable[[], object], repeats: int = 5) -> float:
    """Thời gian huấn luyện (giây), lấy trung vị của ``repeats`` lần chạy."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fit_fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def compare_coefficients(
    feature_names: list[str],
    w_a: np.ndarray, b_a: float,
    w_b: np.ndarray, b_b: float,
    names: tuple[str, str] = ("scratch", "sklearn"),
) -> pd.DataFrame:
    """Bảng so sánh hệ số của hai mô hình và chênh lệch tuyệt đối."""
    df = pd.DataFrame({
        names[0]: np.append(w_a, b_a),
        names[1]: np.append(w_b, b_b),
    }, index=[*feature_names, "bias"])
    df["abs_diff"] = (df[names[0]] - df[names[1]]).abs()
    return df
