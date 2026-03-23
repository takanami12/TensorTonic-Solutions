import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    B, H, W, C = image.shape
    
    # Tính số lượng patch theo chiều dọc và ngang
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    # Reshape để tách biệt các patch
    # (B, H, W, C) -> (B, n_h, patch_size, n_w, patch_size, C)
    patches = image.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C)

    # Hoán vị các trục để gom patch_size, patch_size, C lại với nhau
    # -> (B, n_h, n_w, patch_size, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)

    # Flatten mỗi patch thành một vector
    # -> (B, num_patches, patch_size * patch_size * C)
    patch_embeddings = patches.reshape(B, num_patches, -1)

    # Linear Projection (Chiếu lên không gian embed_dim)
    # Khởi tạo weight ngẫu nhiên thay vì np.ones để đúng bản chất Deep Learning
    weight = np.random.randn(patch_embeddings.shape[-1], embed_dim)
    embedded = patch_embeddings @ weight

    return embedded