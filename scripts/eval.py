import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from patchcore.dataset import MVTecDataset
from patchcore.backbone import get_backbone
from patchcore.model import extract_features
from patchcore.memorybank import MemoryBank
from patchcore.metrics import (
    compute_auroc, compute_max_f1, compute_pro,
    compute_f1_curve, plot_f1_curve,
    compute_confusion_matrix, plot_confusion_matrix,
)
from patchcore.visualization import save_anomaly_visuals
from patchcore.utils import load_config, ensure_dir

def main(cfg_path):
    # Step 1: Load evaluation configuration.
    print(f"[Eval] Loading config: {cfg_path}")
    cfg = load_config(cfg_path)

    # Step 2: Build test dataset and dataloader.
    print("[Eval] Preparing test dataset...")
    dataset = MVTecDataset(
        root=cfg["data"]["root"],
        category=cfg["data"]["category"],
        split="test",
        img_size=cfg["data"]["img_size"],
        train_parquet=cfg["data"].get("train_parquet"),
        test_parquet=cfg["data"].get("test_parquet")
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
    print(f"[Eval] Dataset ready: {len(dataset)} samples in {len(loader)} batches")

    # Step 3: Initialize feature extractor backbone.
    print(f"[Eval] Loading backbone: {cfg['model']['backbone']}")
    backbone = get_backbone(cfg["model"]["backbone"]).to(cfg["device"])
    backbone.eval()

    # Step 4: Load trained memory bank for nearest-neighbor scoring.
    print("[Eval] Loading memory bank...")
    mb = MemoryBank(cfg["model"]["projection_dim"], random_state=cfg["seed"])
    mb.load(os.path.join(cfg["train"]["save_dir"], cfg["data"]["category"], "memorybank.pkl"))
    print("[Eval] Memory bank loaded")

    image_scores, image_labels = [], []
    pixel_scores_all, pixel_labels_all = [], []
    anomaly_maps, gt_masks = [], []

    save_dir = os.path.join(cfg["eval"]["save_dir"], cfg["data"]["category"], "visuals")
    if cfg["eval"]["save_anomaly_maps"]:
        # Optional: Create output directory for anomaly visualizations.
        ensure_dir(save_dir)
        print(f"[Eval] Saving anomaly visuals to: {save_dir}")

    # Step 5: Run inference on each test batch and accumulate scores.
    print("[Eval] Running inference...")
    iterator = loader
    if tqdm is not None:
        # tqdm gives a live progress bar when installed.
        iterator = tqdm(loader, total=len(loader), desc="Eval batches", unit="batch")

    for batch_idx, (imgs, labels, masks, paths) in enumerate(iterator, start=1):
        imgs = imgs.to(cfg["device"])
        feat = extract_features(backbone, imgs, cfg["model"]["layers"])  # B,C,H,W
        B, C, H, W = feat.shape

        for i in range(B):
            f = feat[i].flatten(1).permute(1, 0).cpu().numpy()  # (H*W, C)
            scores = mb.query(f, k=cfg["memorybank"]["knn_k"])
            amap = scores.reshape(H, W)

            amap_resized = cv2.resize(amap, (cfg["data"]["img_size"], cfg["data"]["img_size"]), interpolation=cv2.INTER_LINEAR)
            anomaly_maps.append(amap_resized)
            gt_masks.append(masks[i].numpy())

            image_scores.append(amap_resized.max())
            image_labels.append(labels[i].item())

            pixel_scores_all.append(amap_resized.flatten())
            pixel_labels_all.append(masks[i].flatten())

            if cfg["eval"]["save_anomaly_maps"]:
                base = os.path.basename(paths[i]).rsplit(".", 1)[0]
                save_path = os.path.join(save_dir, base + ".png")
                save_anomaly_visuals(imgs[i].cpu(), amap_resized, save_path, alpha=cfg["eval"]["overlay_alpha"])

            if tqdm is None:
                # Fallback progress message if tqdm is unavailable.
                print(f"[Eval] Batch {batch_idx}/{len(loader)} processed")

    # Step 6: Compute image-level and pixel-level evaluation metrics.
    print("[Eval] Computing metrics...")
    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)
    pixel_scores_all = np.concatenate(pixel_scores_all)
    pixel_labels_all = np.concatenate(pixel_labels_all)

    img_auroc = compute_auroc(image_labels, image_scores)
    pix_auroc = compute_auroc(pixel_labels_all, pixel_scores_all)
    # compute_f1_curve returns full curve data; best_f1/best_t are re-used below.
    img_thresholds, img_f1_vals, img_f1, img_t = compute_f1_curve(
        image_labels, image_scores, cfg["eval"]["num_thresholds"]
    )
    pix_thresholds, pix_f1_vals, pix_f1, pix_t = compute_f1_curve(
        pixel_labels_all, pixel_scores_all, cfg["eval"]["num_thresholds"]
    )
    pro = compute_pro(anomaly_maps, gt_masks, cfg["eval"]["num_thresholds"], cfg["eval"]["pro_fpr_max"])

    print(f"[Image AUROC] {img_auroc:.4f}")
    print(f"[Pixel AUROC] {pix_auroc:.4f}")
    print(f"[Image F1] {img_f1:.4f} @ threshold {img_t:.6f}")
    print(f"[Pixel F1] {pix_f1:.4f} @ threshold {pix_t:.6f}")
    print(f"[PRO] {pro:.4f}")

    # Step 7: Plot and save F1-confidence curves (image-level + pixel-level).
    print("[Eval] Generating F1 curve...")
    f1_save_dir = os.path.join(cfg["eval"]["save_dir"], cfg["data"]["category"])
    f1_path = os.path.join(f1_save_dir, "f1_curve.png")
    plot_f1_curve(
        curves=[
            {"name": "Image-level", "thresholds": img_thresholds,
             "f1_values": img_f1_vals, "best_f1": img_f1, "best_t": img_t,
             "color": "#00b4d8"},
            {"name": "Pixel-level", "thresholds": pix_thresholds,
             "f1_values": pix_f1_vals, "best_f1": pix_f1, "best_t": pix_t,
             "color": "#f72585"},
        ],
        save_path=f1_path,
        title=f"F1-Confidence Curve — {cfg['data']['category']}",
    )
    print(f"[Eval] F1 curve saved: {f1_path}")

    # Step 8: Compute and save confusion matrix at the optimal image-level threshold.
    print("[Eval] Generating confusion matrix...")
    img_preds = (image_scores >= img_t).astype(int)
    cm = compute_confusion_matrix(image_labels, img_preds)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    accuracy  = (tp + tn) / (cm.sum() + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    cm_metrics = {
        "Accuracy":    float(accuracy),
        "Precision":   float(precision),
        "Recall":      float(recall),
        "Specificity": float(specificity),
        "F1":          float(img_f1),
        "Image AUROC": float(img_auroc),
        "Pixel AUROC": float(pix_auroc),
        "PRO":         float(pro),
    }
    cm_save_dir = os.path.join(cfg["eval"]["save_dir"], cfg["data"]["category"])
    cm_path = os.path.join(cm_save_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        cm, cm_path,
        title=f"Confusion Matrix — {cfg['data']['category']}",
        metrics=cm_metrics,
    )
    print(f"[Eval] Confusion matrix saved: {cm_path}")

    # Step 9: Finalize evaluation.
    print("[Eval] Done")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])