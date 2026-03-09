def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top = recommended[:k]
    intersec = set(top).intersection(set(relevant))
    pre = len(intersec) / k
    rec = len(intersec) / len(relevant)
    return [pre, rec]
    