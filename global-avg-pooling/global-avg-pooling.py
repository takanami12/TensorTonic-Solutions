import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    if x.ndim==3:
        C, H, W = x.shape
        return [(1 / (H * W)) * np.sum(x[c]) for c in range(C)]
    else:
        N, C, H, W = x.shape
        return [[(1 / (H * W)) * np.sum(x[n][c]) for c in range(C)] for n in range(N)]