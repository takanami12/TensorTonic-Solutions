import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    if x.ndim==3:
        C, H, W = x.shape
        gaps = []
        for i in range(C):
            gap = (1 / (H * W)) * np.sum(x[i])
            gaps.append(gap)
        return gaps
    else:
        N, C, H, W = x.shape
        gaps_n = []
        for i in range(N):
            gaps_c = []
            for j in range(C):
                gap = (1 / (H * W)) * np.sum(x[i][j])
                gaps_c.append(gap)
            gaps_n.append(gaps_c)
        return gaps_n