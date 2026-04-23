import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    return x.mean(axis=2).mean(axis=1) if x.ndim == 3 else x.mean(axis=3).mean(axis=2)