import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        # YOUR CODE HERE
        ori_shape = x.shape
        x_flattened = x.reshape(self.channels, -1)
        layer1 = relu(self.W1 @ x_flattened)
        layer2 = self.W2 @ layer1
        output_flat = layer2 + x_flattened
        output = output_flat.reshape(ori_shape)
        return output
