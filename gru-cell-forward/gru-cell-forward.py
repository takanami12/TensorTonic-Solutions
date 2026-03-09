import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)

    x, flag = _as2d(x, x.shape[0])
    #Update gate
    z = _sigmoid(x @ params['Wz'] + h_prev @ params['Uz'] + params['bz'])

    #Reset gate
    r = _sigmoid(x @ params['Wr'] + h_prev @ params['Ur'] + params['br'])

    #Candidate Hidden State
    candidate = np.tanh(x @ params['Wh'] + (r * h_prev) @ params['Uh'] + params['bh'])

    #New hidden state
    h = (1 - z) * h_prev + z * candidate

    return h.flatten() if flag else h