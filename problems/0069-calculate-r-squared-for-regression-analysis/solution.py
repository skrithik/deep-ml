import torch

def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the R-squared (R²) coefficient of determination using PyTorch.

    Args:
        y_true (torch.Tensor): Tensor of true values
        y_pred (torch.Tensor): Tensor of predicted values

    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    ssres=torch.sum((y_true-y_pred)**2).float()
    mean=torch.mean(y_true)
    sstot=torch.sum((y_true-mean)**2).float()
    if sstot==0:
        return 0.0
    return (1-(ssres/sstot)).item()