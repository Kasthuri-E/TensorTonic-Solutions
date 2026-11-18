import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    x = x.to(torch.float32)

    if method == "relu":
        return torch.maximum(x, torch.zeros_like(x))

    elif method == "sigmoid":
        return 1.0 / (1.0 + torch.exp(-x))

    elif method == "tanh":
        a = torch.abs(x)
        e = torch.exp(-2.0 * a)
        return torch.sign(x) * (1.0 - e) / (1.0 + e)

    elif method == "leaky_relu":
        return torch.where(x > 0, x, 0.01 * x)

    else:
        raise ValueError(f"unsupported method: {method}")