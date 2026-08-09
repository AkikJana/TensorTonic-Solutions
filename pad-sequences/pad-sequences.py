import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    num_sequences = len(seqs)
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    padded_matrix = np.full((num_sequences, max_len), pad_value)
    for i, seq in enumerate(seqs):
        if len(seq) > max_len:
            padded_matrix[i,] = seq[:max_len]
        else:
            padded_matrix[i, :len(seq)] =  seq
    return padded_matrix