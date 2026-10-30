"""Các hàm vẽ biểu đồ cho EDA và đánh giá mô hình."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FIG_DIR = Path("reports/figures")


def _clean(ax: Axes, grid: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)


def save(fig: Figure, name: str, out_dir: Path = FIG_DIR) -> Path:
    """Lưu hình vào ``reports/figures`` và trả về đường dẫn."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return path


def plot_quality_distribution(quality: pd.Series, threshold: int = 6) -> Figure:
    """Biểu đồ cột phân bố điểm chất lượng, tô màu theo hai lớp."""
    counts = quality.value_counts().sort_index()
    colors = [BLUE if k < threshold else ORANGE for k in counts.index]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    _clean(ax)
    ax.bar(counts.index, counts.values, color=colors, width=0.72)
    for k, v in counts.items():
        ax.text(k, v + 15, str(v), ha="center", fontsize=9)
    ax.axvline(threshold - 0.5, color="#52514e", lw=1, ls="--")
    ax.set_xlabel("Điểm chất lượng")
    ax.set_ylabel("Số mẫu")
    n_pos = int((quality >= threshold).sum())
    ax.set_title(f"Chưa đạt: {len(quality) - n_pos} mẫu | Đạt chuẩn (≥ {threshold}): {n_pos} mẫu", fontsize=10)
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame, label: pd.Series, ncols: int = 4) -> Figure:
    """Boxplot của từng thuộc tính theo hai nhóm nhãn."""
    feats = list(df.columns)
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.4, 1.8 * nrows))
    for ax, col in zip(axs.flat, feats):
        _clean(ax)
        box = ax.boxplot(
            [df.loc[label == 0, col], df.loc[label == 1, col]],
            widths=0.55, patch_artist=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.5),
            medianprops=dict(color="black"),
        )
        for patch, color in zip(box["boxes"], (BLUE, ORANGE)):
            patch.set_facecolor(color)
        ax.set_title(col, fontsize=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Chưa đạt", "Đạt"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for ax in list(axs.flat)[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_corr_heatmap(df: pd.DataFrame, annot_min: float = 0.3) -> Figure:
    """Ma trận tương quan Pearson; chỉ ghi số ở các ô có |r| >= ``annot_min``."""
    corr = df.corr().to_numpy()
    names = list(df.columns)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and abs(corr[i, j]) >= annot_min:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return fig


def plot_loss_curves(histories: dict[str, list[float]], max_iter: int = 300) -> Figure:
    """Đường cong hàm mất mát theo số vòng lặp cho nhiều tốc độ học."""
    fig, ax = plt.subplots(figsize=(6, 3.3))
    _clean(ax)
    for (label, hist), color in zip(histories.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(range(1, len(hist) + 1), hist, lw=2, color=color, label=label)
    ax.set_xlim(0, max_iter)
    ax.set_xlabel("Số vòng lặp")
    ax.set_ylabel("Hàm mất mát J")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: tuple[str, str] = ("Chưa đạt", "Đạt chuẩn")) -> Figure:
    """Ma trận nhầm lẫn dạng ô màu có ghi số."""
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, probas: dict[str, np.ndarray]) -> Figure:
    """Đường ROC của một hoặc nhiều mô hình."""
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    _clean(ax, "both")
    for (name, p), color in zip(probas.items(), (BLUE, ORANGE)):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, lw=2, color=color, label=name)
    ax.plot([0, 1], [0, 1], color="#9a9993", lw=1, ls="--")
    ax.set_xlabel("Tỷ lệ dương giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương thật (TPR)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_c_sweep(l2: pd.DataFrame, l1: pd.DataFrame) -> Figure:
    """Accuracy kiểm định chéo theo C (trục log) cho L2 và L1."""
    fig, ax = plt.subplots(figsize=(6, 3.3))
    _clean(ax)
    for df, label, color in ((l2, "L2 (Ridge)", BLUE), (l1, "L1 (Lasso)", ORANGE)):
        ax.errorbar(df["C"], df["accuracy"] * 100, yerr=df["acc_std"] * 100, marker="o",
                    lw=2, ms=5, capsize=3, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Hệ số C (thang log)")
    ax.set_ylabel("Accuracy kiểm định chéo (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_threshold(table: pd.DataFrame, chosen: float = 0.5) -> Figure:
    """Precision, Recall và F1 theo ngưỡng quyết định."""
    fig, ax = plt.subplots(figsize=(6, 3.3))
    _clean(ax)
    for col, label, color in (("precision", "Precision", BLUE), ("recall", "Recall", ORANGE),
                              ("f1", "F1", AQUA)):
        ax.plot(table["threshold"], table[col], marker="o", ms=4, lw=2, color=color, label=label)
    ax.axvline(chosen, color="#9a9993", ls="--", lw=1)
    ax.set_xlabel("Ngưỡng quyết định")
    ax.set_ylabel("Giá trị")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def plot_odds_ratio(coef: pd.DataFrame) -> Figure:
    """Tỷ số odds (thang log) kèm khoảng tin cậy 95% theo từng đặc trưng."""
    d = coef.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _clean(ax, "x")
    colors = [BLUE if s else "#b8b7b1" for s in d["significant"]]
    ypos = np.arange(len(d))
    ax.hlines(ypos, d["or_low"], d["or_high"], color=colors, lw=2)
    ax.scatter(d["odds_ratio"], ypos, color=colors, zorder=3, s=28)
    ax.axvline(1, color="#9a9993", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d.index)
    ax.set_xscale("log")
    ax.set_xlabel("Tỷ số odds khi tăng 1 độ lệch chuẩn (thang log)")
    fig.tight_layout()
    return fig


def plot_error_by_quality(tbl: pd.DataFrame) -> Figure:
    """Tỷ lệ dự đoán sai theo điểm quality gốc, ghi số mẫu trên mỗi cột."""
    fig, ax = plt.subplots(figsize=(6, 3.3))
    _clean(ax)
    colors = [ORANGE if q <= 5 else BLUE for q in tbl.index]
    ax.bar(tbl.index.astype(str), tbl["error_rate"] * 100, color=colors, width=0.6)
    for i, (rate, n) in enumerate(zip(tbl["error_rate"], tbl["n"])):
        ax.text(i, rate * 100 + 1.2, f"n={n}", ha="center", fontsize=9)
    ax.set_ylim(0, max(tbl["error_rate"]) * 100 + 12)
    ax.set_xlabel("Điểm chất lượng gốc (quality)")
    ax.set_ylabel("Tỷ lệ dự đoán sai (%)")
    fig.tight_layout()
    return fig


def plot_proba_hist(df: pd.DataFrame) -> Figure:
    """Phân bố xác suất dự đoán của hai lớp thực tế."""
    fig, ax = plt.subplots(figsize=(6, 3.3))
    _clean(ax)
    bins = np.linspace(0, 1, 21)
    ax.hist(df.loc[df.y_true == 0, "proba"], bins=bins, alpha=0.75, color=ORANGE, label="Thực tế: chưa đạt")
    ax.hist(df.loc[df.y_true == 1, "proba"], bins=bins, alpha=0.75, color=BLUE, label="Thực tế: đạt chuẩn")
    ax.axvline(0.5, color="#9a9993", ls="--", lw=1)
    ax.set_xlabel("Xác suất dự đoán P(y = 1)")
    ax.set_ylabel("Số mẫu")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    return fig


def plot_learning_curve(tbl: pd.DataFrame) -> Figure:
    """Đường cong học: accuracy huấn luyện và kiểm định chéo theo cỡ mẫu."""
    fig, ax = plt.subplots(figsize=(6, 3.3))
    _clean(ax)
    ax.plot(tbl["n_train"], tbl["train"] * 100, marker="o", ms=4, lw=2, color=ORANGE, label="Huấn luyện")
    ax.plot(tbl["n_train"], tbl["cv"] * 100, marker="o", ms=4, lw=2, color=BLUE, label="Kiểm định chéo")
    ax.set_xlabel("Số mẫu huấn luyện")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig
