import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    if op == "add":
       result = torch.add(x,y)
    elif op == "multiply":
         result = torch.mul(x,y)
    elif op == "power":
         result = torch.pow(x,y)
    elif op == "max":
          result = torch.maximum(x,y)
    elif op == "matmul":
          result = torch.matmul(x,y)
    else:
        raise ValueError(f"unsupported operation:{op}")
    return result.to(torch.float32)                         
    pass
