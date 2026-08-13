"""Generate V10 full-resolution Euclidean and routed fusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from v10_model import ROUTE_NAMES, V10Model

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        a = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(a).view(1, 1, *a.shape).to(device)


def save(x: torch.Tensor, path: Path) -> None:
    a = x.detach().squeeze().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, mode="L").save(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    model = V10Model(
        ckpt["encoder_checkpoint"],
        router_hidden=ckpt.get("router_hidden", 128),
        router_temperature=ckpt.get("router_temperature", 1.0),
        geometry_config=ckpt.get("geometry_config"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    root = Path(args.data_root)
    names = sorted({x.name for x in (root / "ir").iterdir() if x.suffix.lower() in SUFFIXES} & {x.name for x in (root / "vi").iterdir() if x.suffix.lower() in SUFFIXES})
    if args.max_images > 0:
        names = names[:args.max_images]
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_image.jsonl").open("w") as log:
        for i, name in enumerate(names, 1):
            ir, vi = load(root / "ir" / name, device), load(root / "vi" / name, device)
            h, w = ir.shape[-2:]
            ph, pw = (-h) % 32, (-w) % 32
            irp = F.pad(ir, (0, pw, 0, ph), mode="reflect")
            vip = F.pad(vi, (0, pw, 0, ph), mode="reflect")
            with torch.inference_mode():
                e = model.euclidean_forward(irp, vip)[..., :h, :w]
                r, stats = model.routed_forward(irp, vip, return_stats=True)
                r = r[..., :h, :w]
            save(e, out / "euclidean" / name)
            save(r, out / "routed" / name)
            log.write(json.dumps({"dataset": args.dataset, "name": name, "grid_hw": [3, 3], "route_map": stats["route_ids"].reshape(3, 3).cpu().tolist(), "route_fractions": dict(zip(ROUTE_NAMES, stats["route_fractions"].cpu().tolist(), strict=True))}) + "\n")
            log.flush()
            print(json.dumps({"event": "v10_fullres_progress", "image": i, "total": len(names), "name": name}), flush=True)
    (out / "generation_summary.json").write_text(json.dumps({"dataset": args.dataset, "images": len(names), "grid_size": 3, "architecture": "v7_dual_encoder_3x3_hierarchical_euclidean_manifold_router"}, indent=2))


if __name__ == "__main__":
    main()
