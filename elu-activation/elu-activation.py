def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code 
    import math
    return [a if a > 0 else alpha * (math.exp(a) -1) for a in x]