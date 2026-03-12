import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE    
    hidden_states = []
    batch, timestep, _ = X.shape

    for t in range(timestep):
        h_t = np.tanh(h_0 @ W_hh.T + X[:, t, :] @ W_xh.T + b_h)
        hidden_states.append(h_t)
        h_0 = h_t

    res = np.stack(hidden_states, axis=1)

    return res, hidden_states[-1]