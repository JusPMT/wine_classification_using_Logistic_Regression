"""Tải bộ dữ liệu rượu vang đỏ (Wine Quality) từ UCI Machine Learning Repository."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DEFAULT_PATH = Path("data/raw/winequality-red.csv")


def download_red_wine(dest: Path = DEFAULT_PATH, force: bool = False) -> Path:
    """Tải tập tin winequality-red.csv về thư mục ``dest``.

    Args:
        dest: Đường dẫn tập tin đích.
        force: Nếu True thì tải lại kể cả khi tập tin đã tồn tại.

    Returns:
        Đường dẫn tập tin đã lưu.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Đã có sẵn: {dest} (dùng --force để tải lại)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải từ {UCI_URL} ...")
    urllib.request.urlretrieve(UCI_URL, dest)
    print(f"Đã lưu: {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_red_wine(args.dest, args.force)


if __name__ == "__main__":
    main()
