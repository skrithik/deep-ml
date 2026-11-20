import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    x_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    # Your implementation here
    a=torch.matmul(x_t.T,x_t)
    b=torch.matmul(x_t.T,y_t)
    a_i=torch.linalg.inv(a)
    theta=torch.mm(a_i,b)
    return theta
    pass
