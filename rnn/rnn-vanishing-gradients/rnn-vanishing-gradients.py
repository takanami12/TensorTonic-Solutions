import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    gradient_norms = [1] * T
    for t in range(1, T-1):
        gradient_norms[t] *= np.linalg.norm(W_hh, ord=2)
        gradient_norms[t+1] = gradient_norms[t]

    return gradient_norms