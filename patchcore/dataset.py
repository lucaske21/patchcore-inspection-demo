import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", img_size=256):
        self.root = root
        self.category = category
        self.split = split
        self.img_size = img_size
        self.samples = []

        base = os.path.join(root, category, split)
        if split == "train":
            base = os.path.join(base, "good")
            for f in os.listdir(base):
                self.samples.append((os.path.join(base, f), 0, None))
        else:
            for defect in os.listdir(base):
                defect_dir = os.path.join(base, defect)
                for f in os.listdir(defect_dir):
                    label = 0 if defect == "good" else 1
                    mask_path = None
                    if defect != "good":
                        gt_dir = os.path.join(root, category, "ground_truth", defect)
                        mask_name = f.rsplit(".", 1)[0] + "_mask.png"
                        mask_path = os.path.join(gt_dir, mask_name)
                    self.samples.append((os.path.join(defect_dir, f), label, mask_path))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, mask_path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        if mask_path is None:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            m = Image.open(mask_path).convert("L")
            m = self.mask_transform(m)
            mask = (np.array(m) > 0).astype(np.uint8)

        return img, label, mask, path