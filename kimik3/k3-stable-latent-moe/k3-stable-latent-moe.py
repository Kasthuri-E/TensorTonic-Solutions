import torch


def _situ_glu(h: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor,
              down_w: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    """SiTU-GLU expert: down( cap_g*tanh(g/cap_g)*sigmoid(g) * cap_u*tanh(u/cap_u) )."""
    g = h @ gate_w.T                      # [n, inter]
    u = h @ up_w.T                        # [n, inter]
    gate = gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
    up = up_cap * torch.tanh(u / up_cap)
    return (gate * up) @ down_w.T         # [n, out]


def stable_latent_moe(tokens: torch.Tensor, latent_down_projection: torch.Tensor,
                      latent_up_projection: torch.Tensor, router_projection: torch.Tensor,
                      current_bias: torch.Tensor, routed_gate_weights: torch.Tensor,
                      routed_up_weights: torch.Tensor, routed_down_weights: torch.Tensor,
                      shared_gate_weights: torch.Tensor, shared_up_weights: torch.Tensor,
                      shared_down_weights: torch.Tensor, selected_count: int,
                      eps: float = 1e-6, gate_cap: float = 4.0,
                      up_cap: float = 25.0) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing the output, routes, mixture weights, and latent routed aggregate.

    Weight layouts (nn.Linear convention, [out, in]):
      latent_down_projection: [l, d]          latent_up_projection: [d, l]
      router_projection:      [E, d]          current_bias:         [E]
      routed_gate/up_weights: [E, I, l]       routed_down_weights:  [E, l, I]
      shared_gate/up_weights: [2, I_s, d]     shared_down_weights:  [2, d, I_s]
    """
    dtype, device = tokens.dtype, tokens.device
    d = tokens.shape[-1]
    lead_shape = tokens.shape[:-1]
    num_experts = router_projection.shape[0]
    k = int(selected_count)
    if not 1 <= k <= num_experts:
        raise ValueError(f"selected_count must be in [1, {num_experts}], got {k}")
    if shared_gate_weights.shape[0] != 2:
        raise ValueError("Kimi K3 Stable LatentMoE uses exactly two shared experts")

    f32 = lambda t: t.to(device=device, dtype=torch.float32)
    x = f32(tokens.reshape(-1, d))                                   # [N, d]
    n_tokens = x.shape[0]

    # Latent representation z = W_down x
    z = x @ f32(latent_down_projection).T                            # [N, l]
    latent_dim = z.shape[-1]

    # Router: raw sigmoid scores, biased selection, unbiased normalization
    r = torch.sigmoid(x @ f32(router_projection).T)                  # [N, E]
    biased = r + f32(current_bias).reshape(1, num_experts)
    selected = torch.topk(biased, k, dim=-1).indices                  # [N, k]
    sel_scores = r.gather(-1, selected)                               # bias excluded
    weights = sel_scores / sel_scores.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)

    # Routed experts in latent width: u = sum_i p_i E_i(z)
    u = torch.zeros(n_tokens, latent_dim, dtype=torch.float32, device=device)
    for e in torch.unique(selected).tolist():
        tok_idx, slot_idx = (selected == e).nonzero(as_tuple=True)
        expert_out = _situ_glu(z[tok_idx], f32(routed_gate_weights[e]), f32(routed_up_weights[e]),
                               f32(routed_down_weights[e]), gate_cap, up_cap)
        u.index_add_(0, tok_idx, expert_out * weights[tok_idx, slot_idx].unsqueeze(-1))

    # RMSNorm (no learnable scale) before up-projection back to model width
    u_norm = u * torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + eps)
    routed_out = u_norm @ f32(latent_up_projection).T                # [N, d]

    # Two shared experts at model width
    shared_out = sum(
        _situ_glu(x, f32(shared_gate_weights[s]), f32(shared_up_weights[s]),
                  f32(shared_down_weights[s]), gate_cap, up_cap)
        for s in range(2)
    )

    y = shared_out + routed_out

    return {
        "output": y.reshape(*lead_shape, d).to(dtype),
        "selected_experts": selected.reshape(*lead_shape, k).to(torch.long),
        "mixture_weights": weights.reshape(*lead_shape, k).to(dtype),
        "latent_routed_aggregate": u.reshape(*lead_shape, latent_dim).to(dtype),
    }