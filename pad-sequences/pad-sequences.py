import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    max_len = max(len(seq) for seq in seqs) if max_len == None else max_len
    res = np.full((len(seqs), max_len), pad_value)
    for i in range(len(seqs)):
        res[i][0:len(seqs[i])] = seqs[i][:min(len(seqs[i]), max_len)]
    return res