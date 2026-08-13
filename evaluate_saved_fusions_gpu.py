"""Evaluate saved grayscale fusion images with GPU-accelerated tensor metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def entropy(image: np.ndarray) -> float:
    values = np.rint(np.clip(image * 255, 0, 255)).astype(np.uint8)
    counts = np.bincount(values.ravel(), minlength=256).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    x = np.rint(np.clip(a * 255, 0, 255)).astype(np.int32).ravel()
    y = np.rint(np.clip(b * 255, 0, 255)).astype(np.int32).ravel()
    joint = np.bincount(x * 256 + y, minlength=65536).reshape(256, 256).astype(float)
    joint /= joint.sum()
    expected = joint.sum(1, keepdims=True) @ joint.sum(0, keepdims=True)
    valid = joint > 0
    return float((joint[valid] * np.log2(joint[valid] / expected[valid])).sum())


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(float)
    y = b.ravel().astype(float)
    x -= x.mean()
    y -= y.mean()
    return float(np.dot(x, y) / max(np.sqrt(np.dot(x, x) * np.dot(y, y)), 1e-12))


def average_gradient(image: np.ndarray) -> float:
    dx = image[:-1, 1:] - image[:-1, :-1]
    dy = image[1:, :-1] - image[:-1, :-1]
    return float(np.sqrt((dx * dx + dy * dy) / 2).mean() * 255)


def spatial_frequency(image: np.ndarray) -> float:
    row = np.sqrt(np.mean(np.diff(image, axis=0) ** 2))
    column = np.sqrt(np.mean(np.diff(image, axis=1) ** 2))
    return float(np.sqrt(row * row + column * column) * 255)


def qabf(a: np.ndarray, b: np.ndarray, fused: np.ndarray) -> float:
    def gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded = np.pad(image, 1)
        gx = padded[1:-1, 2:] - padded[1:-1, :-2]
        gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
        return np.sqrt(gx * gx + gy * gy + 1e-12), np.arctan2(gy, gx)

    gf, af = gradient(fused)
    qualities, weights = [], []
    for source in (a, b):
        gs, ass = gradient(source)
        ratio = np.minimum(gs, gf) / np.maximum(gs, gf).clip(1e-12)
        angle = 1 - np.remainder(np.abs(ass - af), math.pi) * 2 / math.pi
        qg = 0.9994 / (1 + np.exp(-15 * (ratio - 0.5)))
        qa = 0.9879 / (1 + np.exp(-22 * (angle - 0.8)))
        qualities.append(qg * qa)
        weights.append(gs)
    numerator = (qualities[0] * weights[0] + qualities[1] * weights[1]).sum()
    denominator = np.maximum(weights[0] + weights[1], 1e-12).sum()
    return float(numerator / denominator)


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    window = min(11, *a.shape[-2:])
    window -= int(window % 2 == 0)
    padding = window // 2
    ma = F.avg_pool2d(a, window, 1, padding)
    mb = F.avg_pool2d(b, window, 1, padding)
    va = F.avg_pool2d(a * a, window, 1, padding) - ma * ma
    vb = F.avg_pool2d(b * b, window, 1, padding) - mb * mb
    covariance = F.avg_pool2d(a * b, window, 1, padding) - ma * mb
    cs = (2 * covariance + 0.03**2) / (va + vb + 0.03**2).clamp_min(1e-12)
    return (((2 * ma * mb + 0.01**2) / (ma * ma + mb * mb + 0.01**2).clamp_min(1e-12)) * cs).mean()


def ms_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    weights = a.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    values = []
    for level in range(5):
        values.append(ssim(a, b).clamp_min(1e-6))
        if level < 4:
            a = F.avg_pool2d(a, 2, 2, ceil_mode=True)
            b = F.avg_pool2d(b, 2, 2, ceil_mode=True)
    return torch.prod(torch.stack(values) ** weights)


def vif(reference: torch.Tensor, distorted: torch.Tensor) -> torch.Tensor:
    numerator = reference.new_zeros(())
    denominator = reference.new_zeros(())
    sigma_nsq = 2 / (255**2)
    for scale in range(4):
        if scale:
            reference = F.avg_pool2d(reference, 2, 2, ceil_mode=True)
            distorted = F.avg_pool2d(distorted, 2, 2, ceil_mode=True)
        window = max(3, 17 - 4 * scale)
        window -= int(window % 2 == 0)
        padding = window // 2
        mr = F.avg_pool2d(reference, window, 1, padding)
        md = F.avg_pool2d(distorted, window, 1, padding)
        vr = (F.avg_pool2d(reference * reference, window, 1, padding) - mr * mr).clamp_min(0)
        vd = (F.avg_pool2d(distorted * distorted, window, 1, padding) - md * md).clamp_min(0)
        covariance = F.avg_pool2d(reference * distorted, window, 1, padding) - mr * md
        gain = covariance / (vr + 1e-12)
        noise = vd - gain * covariance
        gain = torch.where(vr < 1e-10, torch.zeros_like(gain), gain)
        noise = torch.where(vr < 1e-10, vd, noise)
        gain = torch.where(vd < 1e-10, torch.zeros_like(gain), gain)
        noise = torch.where(vd < 1e-10, torch.zeros_like(noise), noise).clamp_min(1e-10)
        numerator += torch.log1p(gain * gain * vr / (noise + sigma_nsq)).sum()
        denominator += torch.log1p(vr / sigma_nsq).sum()
    return numerator / denominator.clamp_min(1e-12)


def evaluate(a: np.ndarray, b: np.ndarray, fused: np.ndarray, device: torch.device) -> dict[str, float]:
    tensors = [torch.from_numpy(x.copy()).view(1, 1, *x.shape).to(device) for x in (a, b, fused)]
    ta, tb, tf = tensors
    with torch.inference_mode():
        ssim_total = (ssim(ta, tf) + ssim(tb, tf)) / 2
        ms_ssim_total = (ms_ssim(ta, tf) + ms_ssim(tb, tf)) / 2
        vif_total = vif(ta, tf) + vif(tb, tf)
    mse = (np.mean((fused - a) ** 2) + np.mean((fused - b) ** 2)) / 2
    return {
        "AG": average_gradient(fused),
        "EN": entropy(fused),
        "MI": mutual_information(a, fused) + mutual_information(b, fused),
        "PSNR": -10 * math.log10(max(mse, 1e-12)),
        "Qabf": qabf(a, b, fused),
        "SCD": correlation(fused - a, b) + correlation(fused - b, a),
        "SD": float(np.std(fused * 255, ddof=1)),
        "SF": spatial_frequency(fused),
        "SSIM": float(ssim_total),
        "MS_SSIM_total": float(ms_ssim_total),
        "VIF": float(vif_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--fusion-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    names = sorted({p.stem for p in (args.data_root / "ir").iterdir()} & {p.stem for p in (args.data_root / "vi").iterdir()} & {p.stem for p in args.fusion_root.iterdir()})
    index = lambda directory: {p.stem: p for p in directory.iterdir() if p.is_file()}
    infrared, visible, fused = index(args.data_root / "ir"), index(args.data_root / "vi"), index(args.fusion_root)
    rows = []
    device = torch.device(args.device)
    for number, name in enumerate(names, 1):
        rows.append(evaluate(load_array(infrared[name]), load_array(visible[name]), load_array(fused[name]), device))
        if number % args.progress_every == 0:
            print(json.dumps({"image": number, "total": len(names)}), flush=True)
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    result = {"N": len(rows), "mean": mean}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
