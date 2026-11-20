
import torch

def topk_moe_router(
    x: torch.Tensor,
    w_router: torch.Tensor,
    expert_outputs: torch.Tensor,
    shared_output: torch.Tensor,
    k: int
) -> dict:
    """
    Returns tensors in a dict:
    output, selected_experts, selected_weights,
    token_fraction, probability_mass.
    """
    T = x.shape[0]
    E = w_router.shape[1]
    Dout = expert_outputs.shape[2]

    # Compute router scores: (T, E)
    scores = x @ w_router

    # Select top-k experts with deterministic tie-breaking
    selected_experts = torch.argsort(
        scores, dim=1, descending=True, stable=True
    )[:, :k]

    # Gather selected scores
    selected_scores = torch.gather(
        scores, 1, selected_experts
    )

    # Normalize scores over selected experts
    selected_weights = torch.softmax(
        selected_scores, dim=1
    )

    # Gather corresponding expert outputs
    indices = selected_experts.unsqueeze(-1).expand(
        -1, -1, Dout
    )
    selected_outputs = torch.gather(
        expert_outputs, 1, indices
    )

    # Weighted expert combination + shared output
    output = shared_output + (
        selected_outputs * selected_weights.unsqueeze(-1)
    ).sum(dim=1)

    # Fraction of tokens selecting each expert
    token_fraction = torch.bincount(
        selected_experts.reshape(-1),
        minlength=E
    ).to(x.dtype) / T

    # Mean routing probability across all experts
    probability_mass = torch.softmax(
        scores, dim=1
    ).mean(dim=0)

    return {
        "output": output.to(x.dtype),
        "selected_experts": selected_experts.to(torch.int64),
        "selected_weights": selected_weights.to(x.dtype),
        "token_fraction": token_fraction,
        "probability_mass": probability_mass.to(x.dtype),
    }
