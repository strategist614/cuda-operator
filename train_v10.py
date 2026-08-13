"""Train V10 end-to-end 3x3 hierarchical Euclidean/manifold fusion."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE_CODE = Path("/data/code")
if REMOTE_CODE.is_dir() and str(REMOTE_CODE) not in sys.path:
    sys.path.insert(0, str(REMOTE_CODE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from train_rhcg_unsupervised import TrainConfig, fusion_loss
from train_strong_baseline_geometry import _parse_data_root, build_loader
from v10_model import ROUTE_NAMES, V10Model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-checkpoint", required=True)
    p.add_argument("--data-root", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--samples-per-epoch", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--router-hidden", type=int, default=128)
    p.add_argument("--router-temperature", type=float, default=1.0)
    p.add_argument("--route-balance-weight", type=float, default=0.02)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = TrainConfig(
        data_root=_parse_data_root(args.data_root[0])[1].as_posix(),
        dir_a="ir", dir_b="vi", image_size=args.image_size,
        batch_size=args.batch_size, num_workers=0, seed=args.seed,
        weight_decay=args.weight_decay, val_ratio=0.0,
    )
    loader, manifest = build_loader(cfg, args.data_root, args.max_images, args.samples_per_epoch)
    model = V10Model(
        args.encoder_checkpoint,
        router_hidden=args.router_hidden,
        router_temperature=args.router_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "event": "v10_training_ready", "architecture": "v7_dual_encoder_3x3_hierarchical_router",
        "grid_size": 3, "end_to_end": True, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "datasets": manifest, "samples_per_epoch": len(loader),
    }), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "task": 0.0, "route_balance": 0.0}
        count = 0
        for step, batch in enumerate(loader, 1):
            infrared, visible = batch["a"].to(device), batch["b"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, stats = model.routed_forward(infrared, visible, return_stats=True)
            task, _ = fusion_loss(prediction, infrared, visible, cfg)
            balance = model.routing_regularization(stats)
            loss = task + args.route_balance_weight * balance
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            count += 1
            totals["loss"] += float(loss.detach())
            totals["task"] += float(task.detach())
            totals["route_balance"] += float(balance.detach())
            if step % args.log_interval == 0 or step == len(loader):
                frac = dict(zip(ROUTE_NAMES, stats["route_fractions"].detach().cpu().tolist(), strict=True))
                print(json.dumps({"event": "v10_progress", "epoch": epoch, "step": step, "total_steps": len(loader), "loss": float(loss.detach()), "route_fractions": frac}), flush=True)
        denom = max(count, 1)
        metrics = {f"mean_{k}": v / denom for k, v in totals.items()}
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "metrics": metrics, "grid_size": 3,
            "architecture": "v7_dual_encoder_3x3_hierarchical_euclidean_manifold_router",
            "encoder_checkpoint": str(Path(args.encoder_checkpoint).resolve()),
            "router_hidden": args.router_hidden,
            "router_temperature": args.router_temperature,
            "geometry_config": asdict(model.geometry_config),
            "train_config": asdict(cfg),
        }
        torch.save(checkpoint, output / "last.pt")
        print(json.dumps({"event": "v10_epoch_complete", "epoch": epoch, **metrics}), flush=True)


if __name__ == "__main__":
    main()
