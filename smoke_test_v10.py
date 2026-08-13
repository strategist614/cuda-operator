"""One-step shape, route, gradient, and encoder-initialization check for V10."""

import torch

from v10_model import V10Model


def main() -> None:
    model = V10Model("/data/best_train_loss.pt").cuda().train()
    ir = torch.rand(1, 1, 128, 160, device="cuda")
    vi = torch.rand_like(ir)
    output, stats = model.routed_forward(ir, vi, return_stats=True)
    loss = output.mean() + model.routing_regularization(stats)
    loss.backward()
    print({
        "output_shape": list(output.shape),
        "route_shape": list(stats["route_ids"].shape),
        "grid_hw": list(stats["grid_hw"]),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters_with_grad": sum(p.grad is not None for p in model.parameters()),
        "finite": bool(torch.isfinite(output).all()),
    })


if __name__ == "__main__":
    main()
