import torch

def gini_impurity(y: torch.Tensor) -> float:
    """
    Calculate Gini Impurity for a tensor of class labels.

    :param y: 1D Tensor of class labels (integer type)
    :return: Gini Impurity rounded to three decimal places
    """
    if len(y)==0:
        return 0.0
    unique,counts=torch.unique(y,return_counts=True)
    p=torch.sum((counts.float()/len(y))**2)
    return round((1-p).item(),3)

