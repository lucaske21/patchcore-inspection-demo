import os
import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.measure import label

def compute_auroc(y_true, y_score):
    '''
    Compute the Area Under the Receiver Operating Characteristic Curve (AUROC).
    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_score (np.ndarray): Predicted scores. 
    Returns:
        float: The AUROC score.
    '''
    return roc_auc_score(y_true, y_score)

def compute_max_f1(y_true, y_score, num_thresholds=200):
    '''
    Compute the maximum F1 score for a set of predictions and ground truth labels.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_score (np.ndarray): Predicted scores.
        num_thresholds (int): Number of thresholds to evaluate.

    Returns:
        tuple: The maximum F1 score and the corresponding threshold.
    '''
    thresholds = np.linspace(y_score.min(), y_score.max(), num_thresholds)
    best_f1, best_t = 0.0, thresholds[0]
    for t in thresholds:
        preds = (y_score >= t).astype(np.uint8)
        tp = (preds * y_true).sum()
        fp = (preds * (1 - y_true)).sum()
        fn = ((1 - preds) * y_true).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t

def compute_pro(anomaly_maps, gt_masks, num_thresholds=200, fpr_max=0.3):
    '''
    Compute the Per-Region Overlap (PRO) metric for anomaly detection.

    Args:
        anomaly_maps (list of np.ndarray): List of anomaly maps.
        gt_masks (list of np.ndarray): List of ground truth masks.
        num_thresholds (int): Number of thresholds to evaluate.
        fpr_max (float): Maximum false positive rate for PRO calculation.

    Returns:
        float: The PRO score.
    '''
    thresholds = np.linspace(
        min([m.min() for m in anomaly_maps]),
        max([m.max() for m in anomaly_maps]),
        num_thresholds
    )
    pros, fprs = [], []
    total_neg = sum([(1 - m).sum() for m in gt_masks]) + 1e-8

    for t in thresholds:
        pro_sum, pro_cnt = 0.0, 0
        fp_pixels = 0.0

        for amap, gt in zip(anomaly_maps, gt_masks):
            pred = (amap >= t).astype(np.uint8)
            fp_pixels += (pred * (1 - gt)).sum()

            lab = label(gt)
            for region_id in range(1, lab.max() + 1):
                region = (lab == region_id).astype(np.uint8)
                overlap = (pred * region).sum() / (region.sum() + 1e-8)
                pro_sum += overlap
                pro_cnt += 1

        if pro_cnt == 0:
            continue

        pro = pro_sum / pro_cnt
        fpr = fp_pixels / total_neg
        pros.append(pro)
        fprs.append(fpr)

    # sort by fpr
    fprs, pros = np.array(fprs), np.array(pros)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]

    # clamp fpr <= fpr_max
    mask = fprs <= fpr_max
    if mask.sum() < 2:
        return 0.0

    fprs, pros = fprs[mask], pros[mask]
    # NumPy 2.x removed np.trapz; keep compatibility with both old and new versions.
    trapz_fn = getattr(np, "trapz", None)
    if trapz_fn is None:
        trapz_fn = np.trapezoid
    area = trapz_fn(pros, fprs) / fpr_max
    return area


def compute_f1_curve(y_true, y_score, num_thresholds=200):
    """Compute F1 score at each threshold.

    Returns
    -------
    thresholds : np.ndarray  shape (N,)  — raw score values
    f1_values  : np.ndarray  shape (N,)
    best_f1    : float
    best_t     : float  — threshold at best_f1
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(y_score.min(), y_score.max(), num_thresholds)
    f1_values = np.zeros(num_thresholds)
    for k, t in enumerate(thresholds):
        preds = (y_score >= t).astype(float)
        tp = (preds * y_true).sum()
        fp = (preds * (1 - y_true)).sum()
        fn = ((1 - preds) * y_true).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1_values[k] = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_values))
    return thresholds, f1_values, float(f1_values[best_idx]), float(thresholds[best_idx])


def plot_f1_curve(curves, save_path, title="F1-Confidence Curve"):
    """Plot one or more F1-vs-threshold curves in Ultralytics style.

    Parameters
    ----------
    curves : list of dict, each with keys:
        name        str   — legend label
        thresholds  ndarray
        f1_values   ndarray
        best_f1     float
        best_t      float
        color       str (optional)  — line colour
    save_path : str
    title     : str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG   = "#111827"
    GRID = "#1f2937"
    palette = ["#00b4d8", "#f72585", "#4cc9f0", "#ff9f1c", "#06d6a0"]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Grid ──────────────────────────────────────────────────────
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    # ── Plot each curve ───────────────────────────────────────────
    for i, c in enumerate(curves):
        color = c.get("color", palette[i % len(palette)])
        thresholds = np.asarray(c["thresholds"], dtype=float)
        f1_values  = np.asarray(c["f1_values"],  dtype=float)
        best_f1    = float(c["best_f1"])
        best_t     = float(c["best_t"])

        # Normalise threshold axis to [0, 1] as "confidence" (like Ultralytics)
        t_min, t_max = thresholds.min(), thresholds.max()
        span = t_max - t_min if t_max > t_min else 1.0
        conf = (thresholds - t_min) / span
        best_conf = (best_t - t_min) / span

        ax.plot(conf, f1_values, color=color, linewidth=2.0, zorder=3,
                label=f"{c['name']}  (F1={best_f1:.4f} @ conf={best_conf:.2f})")

        # Peak marker
        ax.plot(best_conf, best_f1, "o", color=color, markersize=7, zorder=4)
        # Vertical dashed line at peak
        ax.axvline(best_conf, color=color, linewidth=0.9, linestyle="--", alpha=0.6, zorder=2)
        # Annotation box
        ax.annotate(
            f"  F1={best_f1:.4f}",
            xy=(best_conf, best_f1),
            xytext=(best_conf + 0.02, best_f1 - 0.06),
            color=color, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    # ── Axes styling ─────────────────────────────────────────────
    ax.set_xlabel("Confidence (normalised threshold)", color="white", fontsize=11, labelpad=8)
    ax.set_ylabel("F1 Score", color="white", fontsize=11, labelpad=8)
    ax.tick_params(colors="white", labelsize=9)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=14)

    legend = ax.legend(
        loc="lower left", fontsize=9,
        facecolor="#1f2937", edgecolor="#4b5563", labelcolor="white",
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return save_path


def compute_roc_curve(y_true, y_score):
    """Compute ROC curve points.

    Returns
    -------
    fprs   : np.ndarray  sorted false-positive rates
    tprs   : np.ndarray  corresponding true-positive rates
    auroc  : float
    """
    from sklearn.metrics import roc_curve
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    fprs, tprs, _ = roc_curve(y_true, y_score)
    auroc = float(roc_auc_score(y_true, y_score))
    return fprs, tprs, auroc


def plot_roc_curve(curves, save_path, title="ROC Curve"):
    """Plot one or more ROC curves in the same dark Ultralytics style.

    Parameters
    ----------
    curves : list of dict, each with keys:
        name   str
        fprs   ndarray
        tprs   ndarray
        auroc  float
        color  str (optional)
    save_path : str
    title     : str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG      = "#111827"
    GRID    = "#1f2937"
    palette = ["#00b4d8", "#f72585", "#4cc9f0", "#ff9f1c", "#06d6a0"]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Grid & diagonal ────────────────────────────────────────────
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280",
            linewidth=1.2, label="Random (AUC=0.50)", zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    # ── Plot each curve ────────────────────────────────────────────
    for i, c in enumerate(curves):
        color = c.get("color", palette[i % len(palette)])
        fprs  = np.asarray(c["fprs"],  dtype=float)
        tprs  = np.asarray(c["tprs"],  dtype=float)
        auroc = float(c["auroc"])

        ax.plot(fprs, tprs, color=color, linewidth=2.5, zorder=3,
                label=f"{c['name']}  (AUC={auroc:.4f})")

        # Mark the point closest to the ideal corner (0,1)
        dist  = np.hypot(fprs, 1.0 - tprs)
        best  = int(np.argmin(dist))
        ax.plot(fprs[best], tprs[best], "o", color=color, markersize=7, zorder=4)
        ax.annotate(
            f"  ({fprs[best]:.3f}, {tprs[best]:.3f})",
            xy=(fprs[best], tprs[best]),
            xytext=(fprs[best] + 0.04, tprs[best] - 0.06),
            color=color, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    # ── Axes styling ───────────────────────────────────────────────
    ax.set_xlabel("False Positive Rate", color="white", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=11, labelpad=8)
    ax.tick_params(colors="white", labelsize=9)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=14)

    ax.legend(
        loc="lower right", fontsize=9,
        facecolor="#1f2937", edgecolor="#4b5563", labelcolor="white",
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return save_path


def compute_confusion_matrix(y_true, y_pred):
    """Return a 2x2 confusion matrix [[TN, FP], [FN, TP]] for binary labels."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix", metrics=None):
    """
    Plot and save a confusion matrix image in Ultralytics style.

    Parameters
    ----------
    cm      : np.ndarray shape (2,2)  [[TN, FP], [FN, TP]]
    save_path : str
    title   : str
    metrics : dict  e.g. {"Precision": 0.95, "Recall": 0.93, ...}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = ["Normal", "Anomaly"]
    n = len(class_names)

    # Row-normalize so each row sums to 1 (recall-perspective).
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # ── Figure layout ──────────────────────────────────────────────
    fig_h = 7.5 if not metrics else 9.0
    fig, ax = plt.subplots(figsize=(8, fig_h))
    BG = "#111827"
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Heatmap ────────────────────────────────────────────────────
    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.ax.set_ylabel("Normalized", color="white", fontsize=9, rotation=270, labelpad=14)

    # ── Cell annotations ───────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            norm_val = cm_norm[i, j]
            count = cm[i, j]
            text_color = "#111827" if norm_val > 0.5 else "white"
            ax.text(j, i, f"{count}\n{norm_val:.1%}",
                    ha="center", va="center",
                    fontsize=15, fontweight="bold", color=text_color)

    # ── Grid lines ─────────────────────────────────────────────────
    for v in np.arange(-0.5, n, 1):
        ax.axhline(v, color="#374151", linewidth=1.0)
        ax.axvline(v, color="#374151", linewidth=1.0)

    # ── Axes labels & title ────────────────────────────────────────
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, color="white", fontsize=13)
    ax.set_yticklabels(class_names, color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.set_xlabel("Predicted Label", color="white", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", color="white", fontsize=12, labelpad=10)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=16)

    # ── Metrics footer ─────────────────────────────────────────────
    if metrics:
        lines = [
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ]
        # 4 items per row
        rows = ["   ".join(lines[i:i + 4]) for i in range(0, len(lines), 4)]
        footer = "\n".join(rows)
        fig.text(
            0.5, 0.01, footer,
            ha="center", va="bottom",
            color="#d1d5db", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1f2937",
                      edgecolor="#4b5563", alpha=0.9)
        )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout(rect=[0, 0.08 if metrics else 0, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return save_path