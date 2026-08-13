"""End-to-end V8 continuous multi-geometry fusion training."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    max_pairs: int = 1500
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    intensity_weight: float = 1.0
    gradient_weight: float = 5.0
    structure_weight: float = 1.0
    mixer_balance_weight: float = 0.01
    gradient_clip: float = 5.0


def paired_names(root: Path) -> list[str]:
    ir_dir = root / "Ir"
    vis_dir = root / "Vis"
    ir_names = {
        path.name for path in ir_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    vis_names = {
        path.name for path in vis_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    names = sorted(ir_names & vis_names)
    if not names:
        raise RuntimeError(f"No paired images found under {root}")
    return names


def load_luminance(path: Path) -> Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class M3FDDataset(Dataset):
    def __init__(self, config: TrainConfig) -> None:
        self.root = Path(config.data_root)
        names = paired_names(self.root)
        rng = random.Random(config.seed)
        self.names = sorted(rng.sample(names, min(config.max_pairs, len(names))))
        self.image_size = config.image_size

    def __len__(self) -> int:
        return len(self.names)

    def _crop(self, infrared: Tensor, visible: Tensor) -> tuple[Tensor, Tensor]:
        size = self.image_size
        _, height, width = infrared.shape
        pad_h = max(0, size - height)
        pad_w = max(0, size - width)
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            infrared = F.pad(infrared, padding, mode="replicate")
            visible = F.pad(visible, padding, mode="replicate")
            _, height, width = infrared.shape
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
        return (
            infrared[:, top:top + size, left:left + size],
            visible[:, top:top + size, left:left + size],
        )

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        name = self.names[index]
        infrared = load_luminance(self.root / "Ir" / name)
        visible = load_luminance(self.root / "Vis" / name)
        infrared, visible = self._crop(infrared, visible)
        if random.random() < 0.5:
            infrared = infrared.flip(-1)
            visible = visible.flip(-1)
        if random.random() < 0.5:
            infrared = infrared.flip(-2)
            visible = visible.flip(-2)
        return {"infrared": infrared, "visible": visible, "name": name}


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class SPDContext(nn.Module):
    def __init__(self, channels: int, rank: int = 24) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels, rank, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(rank * rank, channels),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        reduced = self.reduce(features).flatten(2)
        centered = reduced - reduced.mean(dim=-1, keepdim=True)
        covariance = centered @ centered.transpose(1, 2)
        covariance = covariance / max(1, centered.shape[-1] - 1)
        covariance = covariance + torch.eye(
            covariance.shape[-1], device=features.device, dtype=features.dtype
        ).unsqueeze(0) * 1e-4
        gate = self.gate(covariance.flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return self.project(features * gate)


class HyperbolicContext(nn.Module):
    def __init__(self, channels: int, dimension: int = 32) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, dimension, 1)
        self.up = nn.Conv2d(dimension, channels, 1)
        self.log_curvature = nn.Parameter(torch.zeros(()))

    def forward(self, features: Tensor) -> Tensor:
        tangent = self.down(features)
        curvature = F.softplus(self.log_curvature) + 1e-4
        norm = tangent.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        scale = torch.tanh(curvature.sqrt() * norm) / (curvature.sqrt() * norm)
        return self.up(tangent * scale)


class GrassmannContext(nn.Module):
    def __init__(self, channels: int, dimension: int = 32, rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, dimension, 1)
        self.basis = nn.Parameter(torch.randn(rank, dimension) / math.sqrt(dimension))
        self.up = nn.Conv2d(dimension, channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        latent = F.normalize(self.down(features), dim=1)
        basis = F.normalize(self.basis, dim=1)
        coefficients = torch.einsum("rd,bdhw->brhw", basis, latent)
        projection = torch.einsum("rd,brhw->bdhw", basis, coefficients)
        return self.up(projection)


class ContinuousMixer(nn.Module):
    NAMES = ("SPD", "Hyp", "Gr")

    def __init__(self, initial_gain: float = 0.5) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3))
        self.gain_logit = nn.Parameter(torch.logit(torch.tensor(initial_gain)))

    def forward(self, base: Tensor, residuals: list[Tensor]) -> tuple[Tensor, dict]:
        weights = torch.softmax(self.logits, dim=0)
        mixed = torch.stack(residuals, dim=0)
        mixed = torch.einsum("m,mbchw->bchw", weights, mixed)
        gain = torch.sigmoid(self.gain_logit)
        return base + gain * mixed, {"weights": weights, "gain": gain}

    def balance_loss(self) -> Tensor:
        weights = torch.softmax(self.logits, dim=0)
        uniform = weights.new_full(weights.shape, 1.0 / weights.numel())
        return (weights * (weights.clamp_min(1e-8).log() - uniform.log())).sum()


class EndToEndV8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ir1 = ConvBlock(1, 32)
        self.ir2 = ConvBlock(32, 64, stride=2)
        self.ir3 = ConvBlock(64, 96, stride=2)
        self.vis1 = ConvBlock(1, 32)
        self.vis2 = ConvBlock(32, 64, stride=2)
        self.vis3 = ConvBlock(64, 96, stride=2)
        self.fuse1 = nn.Conv2d(64, 32, 1)
        self.fuse2 = nn.Conv2d(128, 64, 1)
        self.fuse3 = nn.Conv2d(192, 96, 1)
        self.spd = SPDContext(96)
        self.hyperbolic = HyperbolicContext(96)
        self.grassmann = GrassmannContext(96)
        self.mixer = ContinuousMixer()
        self.decode2 = ConvBlock(160, 64)
        self.decode1 = ConvBlock(96, 32)
        self.output = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, infrared: Tensor, visible: Tensor) -> tuple[Tensor, dict]:
        ir1, vis1 = self.ir1(infrared), self.vis1(visible)
        ir2, vis2 = self.ir2(ir1), self.vis2(vis1)
        ir3, vis3 = self.ir3(ir2), self.vis3(vis2)
        skip1 = self.fuse1(torch.cat((ir1, vis1), dim=1))
        skip2 = self.fuse2(torch.cat((ir2, vis2), dim=1))
        base = self.fuse3(torch.cat((ir3, vis3), dim=1))
        mixed, stats = self.mixer(
            base,
            [self.spd(base), self.hyperbolic(base), self.grassmann(base)],
        )
        decoded = F.interpolate(mixed, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode2(torch.cat((decoded, skip2), dim=1))
        decoded = F.interpolate(decoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode1(torch.cat((decoded, skip1), dim=1))
        return torch.sigmoid(self.output(decoded)), stats


def image_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = kernel_x.transpose(0, 1)
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    return F.conv2d(image, kernel_x, padding=1), F.conv2d(image, kernel_y, padding=1)


def ssim(image_a: Tensor, image_b: Tensor, window: int = 11) -> Tensor:
    mu_a = F.avg_pool2d(image_a, window, 1, window // 2)
    mu_b = F.avg_pool2d(image_b, window, 1, window // 2)
    var_a = F.avg_pool2d(image_a.square(), window, 1, window // 2) - mu_a.square()
    var_b = F.avg_pool2d(image_b.square(), window, 1, window // 2) - mu_b.square()
    covariance = F.avg_pool2d(image_a * image_b, window, 1, window // 2) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
    denominator = (mu_a.square() + mu_b.square() + 0.01 ** 2) * (
        var_a + var_b + 0.03 ** 2
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()


def fusion_loss(
    prediction: Tensor,
    infrared: Tensor,
    visible: Tensor,
    model: EndToEndV8,
    config: TrainConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    target = torch.maximum(infrared, visible)
    intensity = F.l1_loss(prediction, target)
    pred_x, pred_y = image_gradients(prediction)
    ir_x, ir_y = image_gradients(infrared)
    vis_x, vis_y = image_gradients(visible)
    target_x = torch.where(ir_x.abs() >= vis_x.abs(), ir_x, vis_x)
    target_y = torch.where(ir_y.abs() >= vis_y.abs(), ir_y, vis_y)
    gradient = F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
    structure = 1.0 - 0.5 * (ssim(prediction, infrared) + ssim(prediction, visible))
    balance = model.mixer.balance_loss()
    total = (
        config.intensity_weight * intensity
        + config.gradient_weight * gradient
        + config.structure_weight * structure
        + config.mixer_balance_weight * balance
    )
    return total, {
        "intensity": intensity,
        "gradient": gradient,
        "structure": structure,
        "mixer_balance": balance,
    }


def save_checkpoint(
    path: Path,
    model: EndToEndV8,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    epoch: int,
    metrics: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
            "architecture": "end_to_end_continuous_multi_geometry_v8",
        },
        path,
    )


def train(config: TrainConfig, max_steps: int = 0) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = M3FDDataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    model = EndToEndV8().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # The covariance branch is numerically fragile in FP16 on Turing GPUs.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps({"pairs": len(dataset), "names": dataset.names}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "training_ready",
        "architecture": "end_to_end_continuous_multi_geometry_v8",
        "device": str(device),
        "pairs": len(dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }), flush=True)
    metrics_path = output_dir / "metrics.jsonl"
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "intensity", "gradient", "structure")}
        processed = 0
        started = time.time()
        for batch in loader:
            infrared = batch["infrared"].to(device, non_blocking=True)
            visible = batch["visible"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=False):
                prediction, stats = model(infrared, visible)
                loss, parts = fusion_loss(prediction, infrared, visible, model, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            processed += 1
            global_step += 1
            totals["loss"] += float(loss.detach())
            for name in ("intensity", "gradient", "structure"):
                totals[name] += float(parts[name].detach())
            if global_step % 25 == 0 or global_step == 1:
                print(json.dumps({
                    "event": "progress",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "mix_weights": dict(zip(
                        model.mixer.NAMES,
                        stats["weights"].detach().float().cpu().tolist(),
                    )),
                    "mix_gain": float(stats["gain"].detach()),
                }), flush=True)
            if max_steps and global_step >= max_steps:
                break
        record = {
            "epoch": epoch,
            **{f"mean_{name}": value / max(1, processed) for name, value in totals.items()},
            "steps": processed,
            "elapsed_seconds": time.time() - started,
            "mix_weights": dict(zip(
                model.mixer.NAMES,
                torch.softmax(model.mixer.logits.detach(), dim=0).cpu().tolist(),
            )),
            "mix_gain": float(torch.sigmoid(model.mixer.gain_logit.detach()).cpu()),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(output_dir / "last.pt", model, optimizer, scaler, config, epoch, record)
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)
        if max_steps and global_step >= max_steps:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/dataset/M3FD")
    parser.add_argument("--output-dir", default="/data/v8/runs/end_to_end_v8_m3fd1500")
    parser.add_argument("--max-pairs", type=int, default=1500)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
