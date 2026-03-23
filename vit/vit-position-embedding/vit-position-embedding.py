import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    # YOUR CODE HERE
    B, N, D = patches.shape
    
    E_pos = np.random.randn(num_patches, embed_dim)

    embedding = patches + E_pos

    return embedding

    

    