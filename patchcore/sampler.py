import numpy as np
from tqdm import tqdm




def kcenter_greedy(X, num_samples):
    """Greedy k-center coreset sampling.

    Each iteration is data-dependent (dist[] must be updated from the
    previous center), so iterations are inherently sequential.
    Speed-up: precompute squared norms and use a BLAS dot-product
    instead of element-wise subtraction + norm, then work in squared
    distance space throughout (argmax and minimum are order-preserving
    under sqrt, so results are identical to the naive version).

    Complexity: O(k * n * d)  — same as before, but ~2-5x faster in
    practice due to cache-friendly matmul vs. broadcast subtraction.
    """
    n = X.shape[0]
    if num_samples >= n:
        return X

    # Precompute ||x_i||^2 for all i  (shape: n,)
    X_sq = (X * X).sum(axis=1)

    first = np.random.randint(0, n)
    centers = [first]
    # Squared distance from every point to the first center
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x·c
    dist_sq = X_sq + X_sq[first] - 2.0 * (X @ X[first])
    np.maximum(dist_sq, 0.0, out=dist_sq)   # numerical safety (sqrt of neg)

    for _ in tqdm(range(num_samples - 1), desc="  k-center sampling", unit="center", ncols=80):
        idx = int(np.argmax(dist_sq))
        centers.append(idx)
        new_dist_sq = X_sq + X_sq[idx] - 2.0 * (X @ X[idx])
        np.maximum(new_dist_sq, 0.0, out=new_dist_sq)
        np.minimum(dist_sq, new_dist_sq, out=dist_sq)

    return X[centers]