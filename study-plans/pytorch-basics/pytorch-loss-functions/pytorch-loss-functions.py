import torch
import torch.nn.functional as F

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    if method == "mse":
        loss = torch.mean((pred - target) ** 2)

    elif method == "cross_entropy":
        loss = F.cross_entropy(pred, target, reduction="mean")

    elif method == "huber":
        loss = F.huber_loss(pred, target, delta=delta, reduction="mean")

    else:
        raise ValueError(f"Unsupported loss method: {method}")

    return loss.item()