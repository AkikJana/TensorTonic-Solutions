import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    num_samples, num_features = X.shape

    # Initialize weights and bias
    w = np.zeros(num_features)
    b = 0.0
    for i in range(steps):
        z = X@w + b
        p = _sigmoid(z)
        w = w - lr * ((np.transpose(X)@(p - y))/len(X))
        b = b - lr * ((np.sum(p - y, axis=0))/len(X))
    return (w, b)
    