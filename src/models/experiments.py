"""Hàm phục vụ thực nghiệm tuần 9–10: kiểm định chéo, ngưỡng, hệ số và phân tích lỗi."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
)

from src.data import preprocess as pp
from src.models.evaluate import compute_metrics
from src.models.logistic_regression_sklearn import build_pipeline

CV_SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")


def load_split_with_quality(
    raw_path=pp.RAW_PATH, threshold: int = pp.QUALITY_THRESHOLD
) -> tuple[pp.SplitData, pd.Series, pd.Series]:
    """Chia dữ liệu như ``preprocess`` nhưng giữ thêm điểm ``quality`` gốc.

    Chỉ số của ``X_train_raw`` / ``X_test_raw`` được dùng để tra lại điểm
    chất lượng, phục vụ phân tích lỗi theo từng mức điểm.

    Returns:
        (SplitData, quality_train, quality_test)
    """
    df = pp.clean(pp.load_raw(raw_path))
    data = pp.split_and_scale(df, threshold=threshold)
    q = df[pp.TARGET_COL]
    return data, q.loc[data.X_train_raw.index], q.loc[data.X_test_raw.index]


def cv_estimate(
    X_raw: pd.DataFrame, y: np.ndarray, C: float = 1.0, k: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Ước lượng trung bình ± độ lệch chuẩn của các thước đo qua k-fold phân tầng."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    res = cross_validate(build_pipeline(C), X_raw, y, cv=cv, scoring=list(CV_SCORING))
    rows = {m: res[f"test_{m}"] for m in CV_SCORING}
    return pd.DataFrame(
        {"mean": {m: v.mean() for m, v in rows.items()},
         "std": {m: v.std(ddof=1) for m, v in rows.items()}}
    )


def c_sweep(
    X_raw: pd.DataFrame, y: np.ndarray, c_values: Sequence[float],
    penalty: str = "l2", k: int = 5, seed: int = 42,
) -> pd.DataFrame:
    """Quét giá trị C (L1 hoặc L2), trả về các thước đo kiểm định chéo và số hệ số khác 0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for c in c_values:
        solver = "lbfgs" if penalty == "l2" else "liblinear"
        clf = LogisticRegression(C=c, penalty=penalty, solver=solver, max_iter=5000)
        pipe = make_pipeline(StandardScaler(), clf)
        with warnings.catch_warnings():
            # tham số `penalty` bị cảnh báo ở scikit-learn >= 1.8 nhưng vẫn chạy đúng
            warnings.simplefilter("ignore")
            res = cross_validate(pipe, X_raw, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
            nnz = int((pipe.fit(X_raw, y)[-1].coef_ != 0).sum())
        rows.append({
            "C": c, "penalty": penalty,
            "accuracy": res["test_accuracy"].mean(), "acc_std": res["test_accuracy"].std(ddof=1),
            "f1": res["test_f1"].mean(), "roc_auc": res["test_roc_auc"].mean(),
            "n_nonzero": nnz,
        })
    return pd.DataFrame(rows)


def oof_probabilities(
    X_raw: pd.DataFrame, y: np.ndarray, C: float = 1.0, k: int = 5, seed: int = 42
) -> np.ndarray:
    """Xác suất dự đoán 'ngoài fold' trên tập huấn luyện (không dùng tập kiểm tra)."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return cross_val_predict(build_pipeline(C), X_raw, y, cv=cv, method="predict_proba")[:, 1]


def threshold_table(
    y: np.ndarray, p: np.ndarray, thresholds: Sequence[float] = np.arange(0.30, 0.76, 0.05)
) -> pd.DataFrame:
    """Các thước đo ứng với từng ngưỡng quyết định."""
    rows = []
    for t in thresholds:
        m = compute_metrics(y, p, threshold=float(t))
        rows.append({"threshold": round(float(t), 2), **{k: m[k] for k in
                     ("accuracy", "precision", "recall", "f1", "FP", "FN")}})
    return pd.DataFrame(rows)


def bootstrap_coefficients(
    X_raw: pd.DataFrame, y: np.ndarray, C: float = 1.0, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Hệ số trên thang z-score, tỷ số odds và khoảng tin cậy 95% bằng bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y)
    X = X_raw.to_numpy()
    coefs = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        pipe = build_pipeline(C).fit(X[idx], y[idx])
        coefs[b] = pipe[-1].coef_[0]
    point = build_pipeline(C).fit(X, y)[-1].coef_[0]
    lo, hi = np.percentile(coefs, [2.5, 97.5], axis=0)
    out = pd.DataFrame({
        "coef": point, "ci_low": lo, "ci_high": hi,
        "odds_ratio": np.exp(point), "or_low": np.exp(lo), "or_high": np.exp(hi),
        "significant": (lo > 0) | (hi < 0),
    }, index=X_raw.columns)
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index)


def build_error_frame(
    X_test_raw: pd.DataFrame, y_true: np.ndarray, proba: np.ndarray,
    quality: pd.Series, threshold: float = 0.5,
) -> pd.DataFrame:
    """Bảng từng mẫu kiểm tra kèm dự đoán, loại lỗi và điểm quality gốc."""
    pred = (proba >= threshold).astype(int)
    df = X_test_raw.copy()
    df["quality"] = quality.to_numpy()
    df["y_true"], df["y_pred"], df["proba"] = y_true, pred, proba
    df["outcome"] = np.select(
        [(y_true == 1) & (pred == 1), (y_true == 0) & (pred == 0),
         (y_true == 0) & (pred == 1)],
        ["TP", "TN", "FP"], default="FN",
    )
    df["margin"] = np.abs(proba - threshold)
    return df


def error_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Tỷ lệ dự đoán sai theo từng điểm quality gốc."""
    g = df.assign(wrong=(df.y_true != df.y_pred).astype(int)).groupby("quality")
    return pd.DataFrame({"n": g.size(), "n_wrong": g.wrong.sum(), "error_rate": g.wrong.mean()})


def error_profile(df: pd.DataFrame, features: list[str], scaler) -> pd.DataFrame:
    """Trung bình đặc trưng (thang z-score) của từng nhóm TP/TN/FP/FN."""
    z = pd.DataFrame(scaler.transform(df[features]), columns=features, index=df.index)
    z["outcome"] = df["outcome"]
    return z.groupby("outcome").mean().T[["TP", "TN", "FP", "FN"]]


def learning_curve_table(X_raw, y, C: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Accuracy huấn luyện và kiểm định chéo theo kích thước tập huấn luyện."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sizes, tr, va = learning_curve(
        build_pipeline(C), X_raw, y, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True, random_state=seed,
    )
    return pd.DataFrame({"n_train": sizes, "train": tr.mean(1), "cv": va.mean(1),
                         "cv_std": va.std(1, ddof=1)})
