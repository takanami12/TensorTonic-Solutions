import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    if X.shape[0] == 0:
        return X
    if strategy == 'mean':
        if X.ndim == 1:
            valid = np.logical_not(np.isnan(X))
            mean = np.mean(X[valid]) if np.any(valid) else 0
            X[np.isnan(X)] = mean
            return X
        else:
            X = X.T
            for x in X:
                valid = np.logical_not(np.isnan(x))
                mean = np.mean(x[valid]) if np.any(valid) else 0
                x[np.isnan(x)] = mean
            return X.T
    else:
        if X.ndim == 1:
            valid = np.logical_not(np.isnan(X))
            median = np.median(X[valid]) if np.any(valid) else 0
            X[np.isnan(X)] = median
            return X
        else:
            X = X.T
            for x in X:
                valid = np.logical_not(np.isnan(x))
                median = np.median(x[valid]) if np.any(valid) else 0
                x[np.isnan(x)] = median
            return X.T