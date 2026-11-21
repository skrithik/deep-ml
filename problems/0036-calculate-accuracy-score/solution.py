import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    n=y_true.shape[0]
    return ((n-np.sum(abs(y_true-y_pred)))/n)
    pass