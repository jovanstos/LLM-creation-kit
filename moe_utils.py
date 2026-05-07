"""
moe_utils.py — Utilities for inspecting MoE routing behaviour.

Use after training (or mid-training) to diagnose expert load imbalance
and measure router confidence. Run with a held-out data split.

Typical workflow:
  1. Train for 10 000+ steps.
  2. Call get_expert_load() on a val DataLoader.
  3. Call print_expert_load_table() — flag any expert < 5% or > 40%.
  4. Call get_router_confidence() — values < 0.4 indicate weak specialisation.
  5. Tune aux_loss_coeff in ModelConfig and retrain if imbalanced.
"""

from collections import defaultdict
from typing import Dict

import torch


# ─── Expert load measurement ──────────────────────────────────────────────────

def get_expert_load(
    model,
    dataloader,
    device,
    n_batches: int = 100,
) -> Dict[int, Dict[int, int]]:
    """
    Count how often each expert is selected across n_batches of data.

    Hooks into every TransformerBlock's MoERouter.forward() to capture
    top_k_indices without modifying the model or its parameters.
    Forward hooks are removed before returning.

    Args:
        model:      MiniLLM instance
        dataloader: yields (x, y) batches
        device:     torch device for inference
        n_batches:  number of batches to sample (more → more accurate counts)

    Returns:
        {layer_idx: {expert_idx: token_slot_count}}
        Token slot count = number of times the expert was chosen across
        all K routing slots (so max is n_tokens * K per batch).
    """
    load: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    hooks = []

    for layer_idx, block in enumerate(model.blocks):
        def make_hook(li):
            def hook_fn(module, inputs, outputs):
                # outputs = (top_k_weights, top_k_indices, aux_loss)
                top_k_indices = outputs[1].detach().cpu()  # (N, K)
                for expert_idx in top_k_indices.flatten().tolist():
                    load[li][int(expert_idx)] += 1
            return hook_fn

        h = block.moe.router.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            model(x.to(device))
    model.train()

    for h in hooks:
        h.remove()

    return {k: dict(v) for k, v in load.items()}


# ─── Display ──────────────────────────────────────────────────────────────────

def print_expert_load_table(load_dict: Dict[int, Dict[int, int]]) -> None:
    """
    Print a formatted table of expert utilisation percentages per layer.

    Flags experts with:
      < 5%  usage — underused, wastes capacity
      > 40% usage — overloaded, may become a bottleneck

    Both conditions indicate load imbalance that aux_loss_coeff should correct.
    Ideal balance for 8 experts, top-2 routing: each expert used ~25% of slots.
    """
    if not load_dict:
        print("No expert load data.")
        return

    n_experts = max(max(d.keys()) for d in load_dict.values()) + 1
    col_w     = 8

    print("\n" + "=" * (8 + n_experts * col_w + 12))
    print("Expert Load Distribution  (% of routed token slots per layer)")
    print("=" * (8 + n_experts * col_w + 12))

    header = f"{'Layer':<8}" + "".join(f"{'Exp' + str(e):>{col_w}}" for e in range(n_experts))
    print(header + "  Status")
    print("-" * (8 + n_experts * col_w + 12))

    for layer_idx in sorted(load_dict.keys()):
        counts = load_dict[layer_idx]
        total  = sum(counts.values())
        if total == 0:
            continue

        fracs  = [100.0 * counts.get(e, 0) / total for e in range(n_experts)]
        row    = f"L{layer_idx:<7}" + "".join(f"{f:{col_w}.1f}%" for f in fracs)

        issues = []
        for e, f in enumerate(fracs):
            if f < 5.0:
                issues.append(f"E{e}<5%")
            elif f > 40.0:
                issues.append(f"E{e}>40%")

        status = f"  WARN({', '.join(issues)})" if issues else "  OK"
        print(row + status)

    print("=" * (8 + n_experts * col_w + 12))
    print("Tip: if imbalanced, reduce/increase aux_loss_coeff in ModelConfig.\n")


# ─── Router confidence ────────────────────────────────────────────────────────

def get_router_confidence(
    model,
    dataloader,
    device,
    n_batches: int = 50,
) -> float:
    """
    Average top-1 routing probability across all tokens and all layers.

    Interpretation:
      > 0.6 — router has learned clear token→expert assignments (healthy)
      0.4–0.6 — moderate specialisation (normal mid-training)
      < 0.4 — router is uncertain; experts not yet differentiated (early training)

    The top-1 weight is taken from the *re-normalised* top-K weights, so it
    represents the dominant expert's share of the combined output, not its
    raw softmax probability.

    Returns:
        Mean top-1 weight as a float in [1/K, 1.0].
    """
    all_probs = []
    hooks     = []

    for block in model.blocks:
        def make_hook():
            def hook_fn(module, inputs, outputs):
                top_k_weights = outputs[0].detach().cpu()  # (N, K) normalised
                all_probs.append(top_k_weights[:, 0])      # dominant expert weight
            return hook_fn

        h = block.moe.router.register_forward_hook(make_hook())
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            model(x.to(device))
    model.train()

    for h in hooks:
        h.remove()

    if not all_probs:
        return 0.0

    confidence = torch.cat(all_probs).mean().item()
    print(f"Router confidence (mean top-1 weight): {confidence:.3f}")
    if confidence < 0.4:
        print("  LOW — experts not yet specialised (normal early in training)")
    elif confidence > 0.6:
        print("  HIGH — experts show clear specialisation")
    else:
        print("  MODERATE — training normally")
    return confidence
