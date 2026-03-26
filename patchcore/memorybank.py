import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import pairwise_distances
import joblib
import os

class MemoryBank:
    def __init__(self, projection_dim=256, random_state=0):
        self.proj = SparseRandomProjection(n_components=projection_dim, random_state=random_state)
        self.features = None

    def fit(self, features: np.ndarray):
        self.features = self.proj.fit_transform(features)

    def query(self, feats: np.ndarray, k=5):
        feats = self.proj.transform(feats)
        dists = pairwise_distances(feats, self.features)
        topk = np.partition(dists, kth=k, axis=1)[:, :k]
        return topk.mean(axis=1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"proj": self.proj, "features": self.features}, path)

    def load(self, path):
        data = joblib.load(path)
        self.proj = data["proj"]
        self.features = data["features"]