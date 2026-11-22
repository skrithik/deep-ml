import torch

def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the precision metric for binary classification.
    
    Precision is defined as the ratio of true positives to the sum of 
    true positives and false positives.
    
    Args:
        y_true: True binary labels (1D tensor)
        y_pred: Predicted binary labels (1D tensor)
    
    Returns:
        Precision value as a scalar tensor
    """
    # Your implementation here
    den=torch.sum(y_pred)
    if den==0 :
        return 0.0
    num=torch.dot(y_pred,y_true)
    return num/den
