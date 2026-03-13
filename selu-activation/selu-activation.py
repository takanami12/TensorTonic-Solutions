import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    # Write code here
    x = np.asarray(x)
    return [np.round(lam * a, 4) if a > 0 else np.round(lam * alpha * (np.exp(a) - 1), 4) for a in x]
