import numpy as np

def shuffle_data(X, y, seed=None):
    # Your code here
    n=X.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(n)
    X_shuff=X[indices]
    Y_shuff=y[indices]
    return X_shuff,Y_shuff
    pass