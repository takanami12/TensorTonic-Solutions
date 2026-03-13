import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    n = len(rater1)
    
    pe = sum([(rater1.tolist().count(label) / n) * (rater2.tolist().count(label) / n) for label in set(rater1.tolist())])
    po = sum(rater1 == rater2) / n

    return (po - pe) / (1 - pe) if pe != 1 else 1