import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    class_nums, class_counts = np.unique(y, return_counts=True)
    probs = class_counts / len(y)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    #print(entropy)
    return entropy