import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from patchcore.dataset import MVTecDataset
from patchcore.backbone import get_backbone
from patchcore.model import extract_features
from patchcore.sampler import kcenter_greedy
from patchcore.memorybank import MemoryBank
from patchcore.utils import load_config, seed_everything, ensure_dir

def main(cfg_path):
    cfg = load_config(cfg_path)
    seed_everything(cfg["seed"])

    dataset = MVTecDataset(
        root=cfg["data"]["root"],
        category=cfg["data"]["category"],
        split="train",
        img_size=cfg["data"]["img_size"]
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])

    backbone = get_backbone(cfg["model"]["backbone"]).to(cfg["device"])
    backbone.eval()

    all_feats = []
    for imgs, _, _, _ in loader:
        imgs = imgs.to(cfg["device"])
        feat = extract_features(backbone, imgs, cfg["model"]["layers"])
        feat = feat.flatten(2).permute(0, 2, 1).reshape(-1, feat.shape[1])
        all_feats.append(feat.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    if cfg["memorybank"]["sampling_method"] == "kcenter":
        all_feats = kcenter_greedy(all_feats, cfg["memorybank"]["max_samples"])

    mb = MemoryBank(cfg["model"]["projection_dim"], random_state=cfg["seed"])
    mb.fit(all_feats)

    save_dir = os.path.join(cfg["train"]["save_dir"], cfg["data"]["category"])
    ensure_dir(save_dir)
    mb.save(os.path.join(save_dir, "memorybank.pkl"))
    print("MemoryBank saved:", os.path.join(save_dir, "memorybank.pkl"))

if __name__ == "__main__":
    import sys
    main(sys.argv[1])