"""Mô hình Hồi quy Logistic của scikit-learn và tìm siêu tham số bằng GridSearchCV."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.001, 0.01, 0.1, 1, 10, 100, 1000)


def build_model(C: float = 1.0, max_iter: int = 5000, tol: float = 1e-4) -> LogisticRegression:
    """Tạo ``LogisticRegression`` (L2, bộ giải L-BFGS).

    ``C = 1`` tương ứng ``lam = 1`` ở bản tự cài đặt.
    """
    return LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=tol)


def build_pipeline(C: float = 1.0, max_iter: int = 5000) -> Pipeline:
    """Pipeline gồm chuẩn hóa và mô hình; dùng khi đưa dữ liệu thô (chưa chuẩn hóa).

    Bộ chuẩn hóa chỉ học tham số trên dữ liệu huấn luyện của mỗi lần ``fit``.
    """
    return make_pipeline(StandardScaler(), build_model(C, max_iter))


def grid_search(
    X_train_raw: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    c_values: Sequence[float] = C_GRID,
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> GridSearchCV:
    """Chọn ``C`` bằng kiểm định chéo phân tầng trên tập huấn luyện.

    Chuẩn hóa nằm trong pipeline nên mỗi fold tự học tham số chuẩn hóa riêng,
    không rò rỉ thông tin giữa các fold.

    Args:
        X_train_raw: Đặc trưng huấn luyện **chưa** chuẩn hóa.
        y_train: Nhãn huấn luyện.
        c_values: Các giá trị C cần thử.
        cv: Số fold.
        scoring: Thước đo dùng để chọn tham số.
        random_state: Hạt giống ngẫu nhiên cho việc chia fold.

    Returns:
        Đối tượng ``GridSearchCV`` đã ``fit``.
    """
    search = GridSearchCV(
        build_pipeline(),
        param_grid={"logisticregression__C": list(c_values)},
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring,
    )
    return search.fit(X_train_raw, y_train)


def grid_results(search: GridSearchCV) -> pd.DataFrame:
    """Bảng kết quả GridSearchCV: C, accuracy trung bình và độ lệch chuẩn."""
    res = search.cv_results_
    return pd.DataFrame({
        "C": [p["logisticregression__C"] for p in res["params"]],
        "mean_score": res["mean_test_score"],
        "std_score": res["std_test_score"],
    })
