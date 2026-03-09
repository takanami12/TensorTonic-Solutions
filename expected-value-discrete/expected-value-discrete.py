import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if sum(p) != 1:
        raise ValueError("ValueError")
    
    return sum(i * j for i, j in zip(x, p)) 
    
