import numpy as np

def simulate_paths(S0, B, T, r, sigma, n, num_paths=50, barrier_type='up'):
    """
    Generates risk-neutral GBM paths for visualization.
    Tracks barrier crossings to compute knockout probability.
    """
    np.random.seed(42)
    dt = T / n
    paths = np.zeros((num_paths, n + 1))
    paths[:, 0] = S0

    for t in range(1, n + 1):
        Z = np.random.randn(num_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Track barrier breaches
    if barrier_type == 'up':
        knocked_out = np.any(paths >= B, axis=1)
    else:
        knocked_out = np.any(paths <= B, axis=1)

    return paths, knocked_out
