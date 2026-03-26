import cv2
import numpy as np
import os

def save_anomaly_visuals(img_tensor, anomaly_map, save_path, alpha=0.5):
    # img_tensor: (3,H,W) [0,1]
    img = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    h, w = img.shape[:2]

    amap = anomaly_map
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heat = (amap_norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path.replace(".png", "_heatmap.png"), heat)
    cv2.imwrite(save_path.replace(".png", "_overlay.png"), overlay)