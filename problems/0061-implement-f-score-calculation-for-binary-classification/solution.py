import torch

def f_score(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float) -> float:
    """
    Calculate F-Score for a binary classification task.

    :param y_true: torch.Tensor of true labels (binary)
    :param y_pred: torch.Tensor of predicted labels (binary)
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    tp=torch.sum((y_true==1) & (y_pred==1))
    fp=torch.sum((y_true==0) & (y_pred==1))
    fn=torch.sum((y_true==1) & (y_pred==0))
    p=0.0
    r=0.0
    if tp+fp!=0:
        p=(tp/(tp+fp)).item()
    if tp+fn!=0:
        r=(tp/(tp+fn)).item()
    if (beta*beta*p+r)==0:
        return 0.0
    return (1+beta*beta)*(p*r/(beta*beta*p+r))
