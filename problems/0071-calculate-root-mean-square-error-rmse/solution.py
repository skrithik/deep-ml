import torch
import torch.nn.functional as F

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between actual and predicted values.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        RMSE value rounded to three decimal places.
    """
    # Write your code here
    loss=torch.mean((y_true-y_pred)**2)

    return torch.sqrt(loss).item()