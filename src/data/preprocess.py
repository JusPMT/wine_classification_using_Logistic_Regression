"""Làm sạch, tạo nhãn, chia tập và chuẩn hóa dữ liệu rượu vang đỏ."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path("data/raw/winequality-red.csv")
PROCESSED_DIR = Path("data/processed")
TARGET_COL = "quality"
QUALITY_THRESHOLD = 6   # quality >= 6 -> nhãn 1 (đạt chuẩn)
TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass
class SplitData:
    """Kết quả tiền xử lý: dữ liệu thô và dữ liệu đã chuẩn hóa."""

    feature_names: list[str]
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: np.ndarray      # đã chuẩn hóa z-score (fit trên tập train)
    X_test: np.ndarray       # chuẩn hóa bằng tham số của tập train
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Đọc tập tin CSV gốc (dấu phân tách là chấm phẩy)."""
    return pd.read_csv(path, sep=";")


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Kiểm tra giá trị thiếu và (tùy chọn) loại bỏ bản ghi trùng lặp.

    Args:
        df: Dữ liệu gốc.
        drop_duplicates: Loại các dòng trùng hoàn toàn nếu True.
    """
    n_missing = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())
    print(f"Số mẫu gốc: {len(df)} | giá trị thiếu: {n_missing} | bản ghi trùng: {n_dup}")
    if n_missing:
        df = df.dropna()
    if drop_duplicates:
        df = df.drop_duplicates()
    return df.reset_index(drop=True)


def make_labels(df: pd.DataFrame, threshold: int = QUALITY_THRESHOLD) -> pd.Series:
    """Nhãn nhị phân: 1 nếu quality >= threshold, ngược lại 0."""
    return (df[TARGET_COL] >= threshold).astype(int)


def split_and_scale(
    df: pd.DataFrame,
    threshold: int = QUALITY_THRESHOLD,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """Chia train/test có phân tầng rồi chuẩn hóa z-score.

    Bộ chuẩn hóa chỉ ``fit`` trên tập train để tránh rò rỉ dữ liệu.
    """
    y = make_labels(df, threshold)
    X = df.drop(columns=TARGET_COL)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler().fit(X_tr)
    return SplitData(
        feature_names=list(X.columns),
        X_train_raw=X_tr,
        X_test_raw=X_te,
        X_train=scaler.transform(X_tr),
        X_test=scaler.transform(X_te),
        y_train=y_tr.to_numpy(),
        y_test=y_te.to_numpy(),
        scaler=scaler,
    )


def save_processed(data: SplitData, out_dir: Path = PROCESSED_DIR) -> None:
    """Lưu dữ liệu đã xử lý vào ``data/processed``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "wine_red_scaled.npz",
        X_train=data.X_train, X_test=data.X_test,
        y_train=data.y_train, y_test=data.y_test,
    )
    train = data.X_train_raw.assign(label=data.y_train)
    test = data.X_test_raw.assign(label=data.y_test)
    train.to_csv(out_dir / "train_raw.csv", index=False)
    test.to_csv(out_dir / "test_raw.csv", index=False)
    params = {
        "feature_names": data.feature_names,
        "mean": data.scaler.mean_.tolist(),
        "scale": data.scaler.scale_.tolist(),
    }
    (out_dir / "scaler_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )


def load_processed(out_dir: Path = PROCESSED_DIR) -> dict[str, np.ndarray]:
    """Đọc lại dữ liệu đã chuẩn hóa từ ``data/processed``."""
    return dict(np.load(out_dir / "wine_red_scaled.npz"))


def run(
    raw_path: Path = RAW_PATH,
    drop_duplicates: bool = True,
    threshold: int = QUALITY_THRESHOLD,
) -> SplitData:
    """Chạy toàn bộ quy trình: đọc, làm sạch, chia, chuẩn hóa, lưu."""
    df = clean(load_raw(raw_path), drop_duplicates=drop_duplicates)
    data = split_and_scale(df, threshold=threshold)
    save_processed(data)
    print(
        f"Sau làm sạch: {len(df)} mẫu | train: {len(data.y_train)} "
        f"({data.y_train.sum()} nhãn 1) | test: {len(data.y_test)} "
        f"({data.y_test.sum()} nhãn 1)"
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Giữ nguyên các bản ghi trùng lặp")
    parser.add_argument("--threshold", type=int, default=QUALITY_THRESHOLD)
    args = parser.parse_args()
    run(args.raw, drop_duplicates=not args.keep_duplicates, threshold=args.threshold)


if __name__ == "__main__":
    main()
