import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """

    if op == "flatten":
       result = torch.flatten(x)
    elif op == "squeeze":
       result = torch.squeeze(x)
    elif op == "transpose":
       result = torch.transpose(x,0,1)
    else:
       raise ValueError(f"unsupported operation:{op}")
    return result.to(torch.float32)
    pass
