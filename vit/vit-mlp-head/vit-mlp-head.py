import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Classification head for ViT.
    """
    # YOUR CODE HERE

    def LayerNorm(x, eps=1e-6):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
        
    # Lấy ra cls token
    cls_token = encoder_output[:, 0, :]

    cls_token = LayerNorm(cls_token)

    d_model = cls_token.shape[-1]

    scale = 1 / np.sqrt(d_model)

    linear_proj = np.random.randn(d_model, num_classes) * scale

    return cls_token @ linear_proj