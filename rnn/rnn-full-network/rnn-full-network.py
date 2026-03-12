import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        output_dim, timestep, input_dim = X.shape

        if h_0 == None:
            h_0 = np.zeros((output_dim, self.hidden_dim))

        y_seq = []

        for t in range(timestep):
            h_t = np.tanh(h_0 @ self.W_hh.T + X[:, t, :] @ self.W_xh.T + self.b_h)
            h_0 = h_t
            y_t = h_t @ self.W_hy.T + self.b_y
            y_seq.append(y_t)

        y_seq = np.stack(y_seq, axis=1)

        return y_seq, h_0