import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    # YOUR CODE HERE

    B = patches.shape[0]
    
    cls = np.random.randn(1, 1, embed_dim)

    cls = np.tile(cls, (B, 1, 1))

    embedded = np.concatenate([cls, patches], axis=1)

    return embedded