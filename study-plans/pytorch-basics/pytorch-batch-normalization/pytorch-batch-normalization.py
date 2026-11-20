import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean = torch.mean(X,dim=0,keepdim=True)

    var = torch.mean((X-mean)**2,dim=0,keepdim=True)
    x_norm = (X-mean)/torch.sqrt(var+eps)
    y = gamma * x_norm + beta

    return y
