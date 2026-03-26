import numpy as np
from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None


def kcenter_greedy_torch(X, num_samples, device="cuda"):
    """Greedy k-center coreset sampling (PyTorch).
    Falls back to CPU if device unavailable.
    Returns numpy array.
    """
    if torch is None:
        raise RuntimeError("PyTorch not available")

    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X)
    else:
        X_t = X

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    X_t = X_t.to(device)
    n = X_t.shape[0]
    if num_samples >= n:
        return X_t.detach().cpu().numpy()

    # Precompute ||x_i||^2
    X_sq = (X_t * X_t).sum(dim=1)

    first = int(torch.randint(0, n, (1,), device=device).item())
    centers = [first]

    dist_sq = X_sq + X_sq[first] - 2.0 * (X_t @ X_t[first])
    dist_sq = torch.clamp(dist_sq, min=0.0)

    for _ in tqdm(range(num_samples - 1), desc="  k-center sampling", unit="center", ncols=80):
        idx = int(torch.argmax(dist_sq).item())
        centers.append(idx)
        new_dist_sq = X_sq + X_sq[idx] - 2.0 * (X_t @ X_t[idx])
        new_dist_sq = torch.clamp(new_dist_sq, min=0.0)
        dist_sq = torch.minimum(dist_sq, new_dist_sq)

    return X_t[centers].detach().cpu().numpy()


def kcenter_greedy(X, num_samples, device="cuda", use_torch=True):
    """Greedy k-center coreset sampling (auto GPU + CPU fallback)."""
    if use_torch and torch is not None:
        try:
            return kcenter_greedy_torch(X, num_samples, device=device)
        except Exception:
            pass

    # ----- CPU numpy fallback -----
    n = X.shape[0]
    if num_samples >= n:
        return X

    X_sq = (X * X).sum(axis=1)

    first = np.random.randint(0, n)
    centers = [first]
    dist_sq = X_sq + X_sq[first] - 2.0 * (X @ X[first])
    np.maximum(dist_sq, 0.0, out=dist_sq)

    for _ in tqdm(range(num_samples - 1), desc="  k-center sampling", unit="center", ncols=80):
        idx = int(np.argmax(dist_sq))
        centers.append(idx)
        new_dist_sq = X_sq + X_sq[idx] - 2.0 * (X @ X[idx])
        np.maximum(new_dist_sq, 0.0, out=new_dist_sq)
        np.minimum(dist_sq, new_dist_sq, out=dist_sq)

    return X[centers]