"""Hồi quy Logistic nhị phân tự cài đặt bằng NumPy (Batch Gradient Descent + L2)."""
from __future__ import annotations

import numpy as np


class LogisticRegressionScratch:
    """Hồi quy Logistic nhị phân, tối ưu bằng Batch Gradient Descent.

    Hàm mất mát: cross-entropy trung bình cộng thêm điều chuẩn L2
    ``lam / (2n) * ||w||^2``. Hệ số chặn ``b`` không bị điều chuẩn.
    Với ``lam = 1 / C`` thì mô hình tương đương
    ``sklearn.linear_model.LogisticRegression(C=C)``.

    Args:
        lr: Tốc độ học.
        n_iter: Số vòng lặp tối đa.
        lam: Hệ số điều chuẩn L2 (lambda).
        tol: Dừng khi mức giảm của hàm mất mát nhỏ hơn ``tol``.

    Attributes:
        w: Vectơ trọng số, kích thước (d,).
        b: Hệ số chặn.
        history: Giá trị hàm mất mát qua từng vòng lặp.
    """

    def __init__(
        self,
        lr: float = 0.5,
        n_iter: int = 20000,
        lam: float = 1.0,
        tol: float = 1e-10,
    ) -> None:
        self.lr, self.n_iter, self.lam, self.tol = lr, n_iter, lam, tol
        self.w: np.ndarray = np.zeros(0)
        self.b: float = 0.0
        self.history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid ổn định số học: tránh tràn số khi |z| lớn."""
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _loss(self, p: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy trung bình cộng thành phần điều chuẩn L2."""
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        n = len(y)
        return float(ce.mean() + self.lam / (2 * n) * np.sum(self.w ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Huấn luyện mô hình trên (X, y); y nhận giá trị 0 hoặc 1."""
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        self.history = []
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            grad_w = X.T @ err / n + self.lam / n * self.w
            grad_b = err.mean()          # hệ số chặn không bị điều chuẩn
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.history.append(self._loss(p, y))
            if (
                len(self.history) > 1
                and abs(self.history[-2] - self.history[-1]) < self.tol
            ):
                break
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Xác suất mẫu thuộc lớp 1."""
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Nhãn dự đoán: 1 nếu xác suất >= ``threshold``."""
        return (self.predict_proba(X) >= threshold).astype(int)
