import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.asarray(x)
    return x * _sigmoid(x)