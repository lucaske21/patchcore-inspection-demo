import numpy as np

def kcenter_greedy(X, num_samples):
    n = X.shape[0]
    centers = [np.random.randint(0, n)]
    dist = np.linalg.norm(X - X[centers[0]], axis=1)

    for _ in range(num_samples - 1):
        idx = np.argmax(dist)
        centers.append(idx)
        dist = np.minimum(dist, np.linalg.norm(X - X[idx], axis=1))
    return X[centers]