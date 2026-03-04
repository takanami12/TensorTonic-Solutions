import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)
    
    if rng is None:
        rng = np.random
        
    random_values = rng.random(x.shape)
    
    dropout_pattern = (random_values < (1 - p)).astype(np.float32)
    
    output = x * dropout_pattern / (1 - p)
    
    return output, dropout_pattern / (1 - p)