import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Your code here
    n=X.shape[0]
    y_pred=np.matmul(X,w)
    loss=(1/n)*(np.sum((y_true-y_pred)**2))+alpha*(np.sum(w**2))
    return loss
