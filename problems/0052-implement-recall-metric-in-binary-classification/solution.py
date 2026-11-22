import torch

def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Tensor of true binary labels (0 or 1)
        y_pred: Tensor of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    tp=torch.sum((y_true==1) & (y_pred==1))
    fn=torch.sum((y_true==1) & (y_pred==0))
    if tp+fn==0:
        return 0.0
    return (tp/(tp+fn)).item()