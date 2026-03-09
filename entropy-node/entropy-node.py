import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    labels = list(set(y))
    prob = {label: y.count(label)/len(y) for label in labels}
    return -sum([p * np.log2(p) for p in prob.values()]) if len(y) > 0 else 0.0