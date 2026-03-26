import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.measure import label

def compute_auroc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

def compute_max_f1(y_true, y_score, num_thresholds=200):
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