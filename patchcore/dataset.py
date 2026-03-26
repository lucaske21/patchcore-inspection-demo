import os
import io
import base64
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import pandas as pd

class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", img_size=256, train_parquet=None, test_parquet=None):
        self.root = root
        self.category = category
        self.split = split
        self.img_size = img_size
        self.samples = []
        self.use_parquet = bool(train_parquet or test_parquet)

        if self.use_parquet:
            parquet_path = train_parquet if split == "train" else test_parquet
            if parquet_path is None:
                raise ValueError(f"parquet path is required for split={split}")
            self.samples = self._load_from_parquet(parquet_path)
        else:
            self.samples = self._load_from_folders()

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST)
        ])

    def _load_from_folders(self):
        samples = []
        base = os.path.join(self.root, self.category, self.split)
        if self.split == "train":
            base = os.path.join(base, "good")
            for f in os.listdir(base):
                samples.append((os.path.join(base, f), 0, None, False))
        else:
            for defect in os.listdir(base):
                defect_dir = os.path.join(base, defect)
                for f in os.listdir(defect_dir):
                    label = 0 if defect == "good" else 1
                    mask_path = None
                    if defect != "good":
                        gt_dir = os.path.join(self.root, self.category, "ground_truth", defect)
                        mask_name = f.rsplit(".", 1)[0] + "_mask.png"
                        mask_path = os.path.join(gt_dir, mask_name)
                    samples.append((os.path.join(defect_dir, f), label, mask_path, False))
        return samples

    def _load_from_parquet(self, parquet_path):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"parquet file not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        image_col = self._pick_col(df.columns, ["image", "img", "input", "pixel_values"])
        if image_col is None:
            raise ValueError(f"cannot find image column in parquet: {parquet_path}")

        label_col = self._pick_col(df.columns, ["label", "is_anomaly", "anomaly", "target", "y"])
        mask_col = self._pick_col(df.columns, ["mask", "segmentation_mask", "anomaly_mask", "gt_mask"])
        path_col = self._pick_col(df.columns, ["path", "image_path", "file_name", "filename", "id"])

        samples = []
        for idx, row in df.iterrows():
            raw_image = row[image_col]
            label = 0
            if label_col is not None:
                label = int(bool(row[label_col]))
            elif self.split == "test":
                label = 0

            raw_mask = row[mask_col] if mask_col is not None else None
            sample_path = str(row[path_col]) if path_col is not None else f"{self.category}_{self.split}_{idx}"
            samples.append((raw_image, label, raw_mask, True, sample_path))
        return samples

    @staticmethod
    def _pick_col(columns, candidates):
        col_map = {str(c).lower(): c for c in columns}
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    @staticmethod
    def _decode_image(raw):
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, dict) and "bytes" in raw:
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if isinstance(raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(raw, str):
            if os.path.exists(raw):
                return Image.open(raw).convert("RGB")
            try:
                return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            except Exception as e:
                raise ValueError("string image is neither path nor base64") from e
        if isinstance(raw, np.ndarray):
            arr = raw
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L").convert("RGB")
            return Image.fromarray(arr).convert("RGB")
        raise ValueError(f"unsupported image type: {type(raw)}")

    @staticmethod
    def _decode_mask(raw, img_size):
        if raw is None:
            return np.zeros((img_size, img_size), dtype=np.uint8)
        if isinstance(raw, Image.Image):
            m = raw.convert("L")
        elif isinstance(raw, dict) and "bytes" in raw:
            m = Image.open(io.BytesIO(raw["bytes"])).convert("L")
        elif isinstance(raw, (bytes, bytearray)):
            m = Image.open(io.BytesIO(raw)).convert("L")
        elif isinstance(raw, str):
            if os.path.exists(raw):
                m = Image.open(raw).convert("L")
            else:
                m = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
        elif isinstance(raw, np.ndarray):
            arr = raw.astype(np.uint8)
            if arr.ndim == 3:
                arr = arr[..., 0]
            m = Image.fromarray(arr, mode="L")
        else:
            return np.zeros((img_size, img_size), dtype=np.uint8)
        m = m.resize((img_size, img_size), resample=Image.NEAREST)
        return (np.array(m) > 0).astype(np.uint8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.use_parquet:
            raw_image, label, raw_mask, _, sample_path = sample
            img = self._decode_image(raw_image)
        else:
            path, label, mask_path, _ = sample
            sample_path = path
            raw_mask = mask_path
            img = Image.open(path).convert("RGB")
        img = self.transform(img)

        if self.use_parquet:
            mask = self._decode_mask(raw_mask, self.img_size)
        elif raw_mask is None:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            m = Image.open(raw_mask).convert("L")
            m = self.mask_transform(m)
            mask = (np.array(m) > 0).astype(np.uint8)

        return img, int(label), mask, sample_path