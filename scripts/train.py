import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from patchcore.dataset import MVTecDataset
from patchcore.backbone import get_backbone
from patchcore.model import extract_features
from patchcore.sampler import kcenter_greedy
from patchcore.memorybank import MemoryBank
from patchcore.utils import load_config, seed_everything, ensure_dir

TOTAL_STEPS = 5

def step(n, desc):
    print(f"\n[{n}/{TOTAL_STEPS}] {desc}")

def main(cfg_path):
    step(1, "Loading config & dataset")
    cfg = load_config(cfg_path)
    seed_everything(cfg["seed"])

    dataset = MVTecDataset(
        root=cfg["data"]["root"],
        category=cfg["data"]["category"],
        split="train",
        img_size=cfg["data"]["img_size"],
        train_parquet=cfg["data"].get("train_parquet"),
        test_parquet=cfg["data"].get("test_parquet")
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    print(f"    Dataset size: {len(dataset)} images  |  Batches: {len(loader)}")

    step(2, "Loading backbone")
    backbone = get_backbone(cfg["model"]["backbone"]).to(cfg["device"])
    backbone.eval()
    print(f"    Backbone: {cfg['model']['backbone']}  |  Device: {cfg['device']}")

    step(3, "Extracting patch features")
    all_feats = []
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(loader, desc="  feature extraction", unit="batch", ncols=80):
            imgs = imgs.to(cfg["device"])
            feat = extract_features(backbone, imgs, cfg["model"]["layers"])
            feat = feat.flatten(2).permute(0, 2, 1).reshape(-1, feat.shape[1])
            all_feats.append(feat.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    print(f"    Raw feature matrix: {all_feats.shape}")

    step(4, f"Subsampling memory bank ({cfg['memorybank']['sampling_method']})")
    if cfg["memorybank"]["sampling_method"] == "kcenter":
        all_feats = kcenter_greedy(all_feats, cfg["memorybank"]["max_samples"])
    print(f"    Sampled feature matrix: {all_feats.shape}")

    step(5, "Fitting & saving memory bank")
    mb = MemoryBank(cfg["model"]["projection_dim"], random_state=cfg["seed"])
    mb.fit(all_feats)

    save_dir = os.path.join(cfg["train"]["save_dir"], cfg["data"]["category"])
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, "memorybank.pkl")
    mb.save(save_path)
    print(f"\n✓ Done. MemoryBank saved: {save_path}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])