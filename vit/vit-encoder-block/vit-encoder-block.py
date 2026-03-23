import numpy as np

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> np.ndarray:
    """
    ViT Transformer encoder block.
    """
    # Architecture:
    # x' = x + MSA(LN(x))
    # x'' = x' + MLP(LN(x'))
    
    # YOUR CODE HERE
    def GELU(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x ** 3)))

    def LayerNorm(x):
        # Chuẩn hóa theo chiều axis=-1
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / std

    def MSA(x, num_heads, embed_dim):
        B, N, C = x.shape
        # Tính số chiều của mỗi head
        head_dims = embed_dim // num_heads

        # Tính hệ số k trong công thức của attention
        scale = 1 / np.sqrt(head_dims)

        # Khởi tạo 3 ma trận q, k, v để thực hiện attention
        W_q = np.random.randn(C, C)
        W_k = np.random.randn(C, C)
        W_v = np.random.randn(C, C)

        # Tính toán các vector q, k, v
        q = x @ W_q
        k = x @ W_k
        v = x @ W_v

        # Tách ra để đưa vào từng head: (B, N, num_heads, head_dims) -> (B, num_heads, N, head_dims)
        q = q.reshape(B, N, num_heads, head_dims).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, head_dims).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, num_heads, head_dims).transpose(0, 2, 1, 3)

        # Tính toán attention weights
        attention_logits = (q @ k.transpose(0, 1, 3, 2)) * scale

        e_x = np.exp(attention_logits - np.max(attention_logits, axis=-1, keepdims=True))

        attention_weights = e_x / e_x.sum(axis=-1, keepdims=True)
        
        out = (attention_weights @ v).transpose(0, 2, 1, 3).reshape(B, N, C)

        W_o = np.random.randn(C, C)

        return out @ W_o

    def MLP(x, mlp_ratio):
        hidden_dims = int(embed_dim * mlp_ratio)
        W1 = np.random.randn(embed_dim, hidden_dims)
        W2 = np.random.randn(hidden_dims, embed_dim)

        return GELU(x @ W1) @ W2

    res1 = x
    x = LayerNorm(x)
    x = MSA(x, num_heads, embed_dim)
    x = res1 + x

    res2 = x
    x = LayerNorm(x)
    x = MLP(x, mlp_ratio)
    x = res2 + x

    return x
        
        

        
        
        
        
        