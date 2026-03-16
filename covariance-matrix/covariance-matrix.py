import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    
    mu = np.mean(X, axis=0)
    X_cen = X - mu

    if X.ndim == 1 or X.shape[0] == 1:
        return None
    else:
        cov = 1 / (X.shape[0] - 1) * X_cen.T @ X_cen
        return cov