import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from patchcore.dataset import MVTecDataset
from patchcore.backbone import get_backbone
from patchcore.model import extract_features
from patchcore.memorybank import MemoryBank
from patchcore.metrics import compute_auroc, compute_max_f1, compute_pro
from patchcore.visualization import save_anomaly_visuals
from patchcore.utils import load_config, ensure_dir

def main(cfg_path):
    cfg = load_config(cfg_path)

    dataset = MVTecDataset(
        root=cfg["data"]["root"],
        category=cfg["data"]["category"],
        split="test",
        img_size=cfg["data"]["img_size"]
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    backbone = get_backbone(cfg["model"]["backbone"]).to(cfg["device"])
    backbone.eval()

    mb = MemoryBank(cfg["model"]["projection_dim"], random_state=cfg["seed"])
    mb.load(os.path.join(cfg["train"]["save_dir"], cfg["data"]["category"], "memorybank.pkl"))

    image_scores, image_labels = [], []
    pixel_scores_all, pixel_labels_all = [], []
    anomaly_maps, gt_masks = [], []

    save_dir = os.path.join(cfg["eval"]["save_dir"], cfg["data"]["category"], "visuals")
    if cfg["eval"]["save_anomaly_maps"]:
        ensure_dir(save_dir)

    for imgs, labels, masks, paths in loader:
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

    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)
    pixel_scores_all = np.concatenate(pixel_scores_all)
    pixel_labels_all = np.concatenate(pixel_labels_all)

    img_auroc = compute_auroc(image_labels, image_scores)
    pix_auroc = compute_auroc(pixel_labels_all, pixel_scores_all)
    img_f1, img_t = compute_max_f1(image_labels, image_scores, cfg["eval"]["num_thresholds"])
    pix_f1, pix_t = compute_max_f1(pixel_labels_all, pixel_scores_all, cfg["eval"]["num_thresholds"])
    pro = compute_pro(anomaly_maps, gt_masks, cfg["eval"]["num_thresholds"], cfg["eval"]["pro_fpr_max"])

    print(f"[Image AUROC] {img_auroc:.4f}")
    print(f"[Pixel AUROC] {pix_auroc:.4f}")
    print(f"[Image F1] {img_f1:.4f} @ threshold {img_t:.6f}")
    print(f"[Pixel F1] {pix_f1:.4f} @ threshold {pix_t:.6f}")
    print(f"[PRO] {pro:.4f}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])