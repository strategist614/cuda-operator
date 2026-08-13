"""Run full-resolution V8 fusion and standard fusion metrics on datasets."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_v8_end_to_end import EndToEndV8


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATASETS = {
    "TNO": ("ir", "vi"),
    "RoadScene": ("cropinfrared", "crop_LR_visible"),
    "MSRS": ("test/ir", "test/vi"),
    "FMB": ("test/Infrared", "test/Visible"),
    "M3FD": ("Ir", "Vis"),
}
METRIC_NAMES = (
    "AG", "EN", "MI", "PSNR", "Qabf", "SCD", "SD", "SF", "SSIM",
    "MS_SSIM_total", "VIF",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-base", default="/data/dataset")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def indexed_images(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def paired_images(root: Path, ir_subdir: str, vi_subdir: str) -> list[tuple[str, Path, Path]]:
    infrared = indexed_images(root / ir_subdir)
    visible = indexed_images(root / vi_subdir)
    names = sorted(infrared.keys() & visible.keys())
    if not names:
        raise RuntimeError(f"No image pairs found in {root}")
    return [(name, infrared[name], visible[name]) for name in names]


def load_luminance(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).view(1, 1, *array.shape).to(device)


def save_image(image: torch.Tensor, path: Path) -> None:
    array = image.squeeze().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def entropy(image: np.ndarray) -> float:
    values = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def mutual_information(source: np.ndarray, fused: np.ndarray) -> float:
    a = np.clip(np.rint(source * 255), 0, 255).astype(np.int32).ravel()
    b = np.clip(np.rint(fused * 255), 0, 255).astype(np.int32).ravel()
    joint = np.bincount(a * 256 + b, minlength=65536).reshape(256, 256).astype(np.float64)
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    expected = pa @ pb
    valid = joint > 0
    return float((joint[valid] * np.log2(joint[valid] / expected[valid])).sum())


def average_gradient(image: np.ndarray) -> float:
    dx = image[:-1, 1:] - image[:-1, :-1]
    dy = image[1:, :-1] - image[:-1, :-1]
    return float(np.sqrt((dx ** 2 + dy ** 2) / 2).mean() * 255)


def spatial_frequency(image: np.ndarray) -> float:
    rf = np.sqrt(np.mean(np.diff(image, axis=0) ** 2))
    cf = np.sqrt(np.mean(np.diff(image, axis=1) ** 2))
    return float(np.sqrt(rf * rf + cf * cf) * 255)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / max(denominator, 1e-12))


def ssim_value(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    window = min(11, a.shape[-2], a.shape[-1])
    if window % 2 == 0:
        window -= 1
    padding = window // 2
    mu_a = F.avg_pool2d(a, window, 1, padding)
    mu_b = F.avg_pool2d(b, window, 1, padding)
    var_a = F.avg_pool2d(a * a, window, 1, padding) - mu_a * mu_a
    var_b = F.avg_pool2d(b * b, window, 1, padding) - mu_b * mu_b
    cov = F.avg_pool2d(a * b, window, 1, padding) - mu_a * mu_b
    cs = (2 * cov + 0.03 ** 2) / (var_a + var_b + 0.03 ** 2).clamp_min(1e-12)
    ss = ((2 * mu_a * mu_b + 0.01 ** 2) / (mu_a * mu_a + mu_b * mu_b + 0.01 ** 2).clamp_min(1e-12)) * cs
    return ss.mean()


def ms_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    weights = a.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    values = []
    for level in range(5):
        values.append(ssim_value(a, b).clamp_min(1e-6))
        if level < 4:
            a = F.avg_pool2d(a, 2, 2, ceil_mode=True)
            b = F.avg_pool2d(b, 2, 2, ceil_mode=True)
    return torch.prod(torch.stack(values) ** weights)


def qabf(source_a: torch.Tensor, source_b: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
    kx = fused.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)

    def gradient(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = F.conv2d(image, kx, padding=1)
        gy = F.conv2d(image, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12), torch.atan2(gy, gx)

    gf, af = gradient(fused)

    def quality(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gs, ass = gradient(source)
        ratio = torch.minimum(gs, gf) / torch.maximum(gs, gf).clamp_min(1e-12)
        angle = 1 - (ass - af).abs().remainder(math.pi) * 2 / math.pi
        qg = 0.9994 / (1 + torch.exp(-15 * (ratio - 0.5)))
        qa = 0.9879 / (1 + torch.exp(-22 * (angle - 0.8)))
        return qg * qa, gs

    qa, ga = quality(source_a)
    qb, gb = quality(source_b)
    weights_a = ga.pow(1.0)
    weights_b = gb.pow(1.0)
    return (qa * weights_a + qb * weights_b).sum() / (weights_a + weights_b).sum().clamp_min(1e-12)


def vif_single(reference: torch.Tensor, distorted: torch.Tensor) -> torch.Tensor:
    # Four-scale pixel-domain VIF, matching the common vifp_mscale definition.
    numerator = reference.new_zeros(())
    denominator = reference.new_zeros(())
    sigma_nsq = 2.0 / (255.0 ** 2)
    ref, dist = reference, distorted
    for scale in range(4):
        if scale:
            ref = F.avg_pool2d(ref, 2, 2, ceil_mode=True)
            dist = F.avg_pool2d(dist, 2, 2, ceil_mode=True)
        window = max(3, 17 - 4 * scale)
        if window % 2 == 0:
            window -= 1
        pad = window // 2
        mu_r = F.avg_pool2d(ref, window, 1, pad)
        mu_d = F.avg_pool2d(dist, window, 1, pad)
        var_r = (F.avg_pool2d(ref * ref, window, 1, pad) - mu_r * mu_r).clamp_min(0)
        var_d = (F.avg_pool2d(dist * dist, window, 1, pad) - mu_d * mu_d).clamp_min(0)
        cov = F.avg_pool2d(ref * dist, window, 1, pad) - mu_r * mu_d
        gain = cov / (var_r + 1e-12)
        noise = var_d - gain * cov
        gain = torch.where(var_r < 1e-10, torch.zeros_like(gain), gain)
        noise = torch.where(var_r < 1e-10, var_d, noise)
        gain = torch.where(var_d < 1e-10, torch.zeros_like(gain), gain)
        noise = torch.where(var_d < 1e-10, torch.zeros_like(noise), noise)
        noise = noise.clamp_min(1e-10)
        numerator += torch.log1p(gain * gain * var_r / (noise + sigma_nsq)).sum()
        denominator += torch.log1p(var_r / sigma_nsq).sum()
    return numerator / denominator.clamp_min(1e-12)


def metrics(ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> dict[str, float]:
    ir_np = ir.squeeze().cpu().numpy()
    vi_np = vi.squeeze().cpu().numpy()
    fu_np = fused.squeeze().cpu().numpy()
    mse_ir = float(np.mean((fu_np - ir_np) ** 2))
    mse_vi = float(np.mean((fu_np - vi_np) ** 2))
    with torch.no_grad():
        ssim = (ssim_value(ir, fused) + ssim_value(vi, fused)) / 2
        msssim = (ms_ssim(ir, fused) + ms_ssim(vi, fused)) / 2
        q = qabf(ir, vi, fused)
        vif = vif_single(ir, fused) + vif_single(vi, fused)
    return {
        "AG": average_gradient(fu_np),
        "EN": entropy(fu_np),
        "MI": mutual_information(ir_np, fu_np) + mutual_information(vi_np, fu_np),
        "PSNR": (-10 * math.log10(max((mse_ir + mse_vi) / 2, 1e-12))),
        "Qabf": float(q),
        "SCD": correlation(fu_np - ir_np, vi_np) + correlation(fu_np - vi_np, ir_np),
        "SD": float(np.std(fu_np * 255, ddof=1)),
        "SF": spatial_frequency(fu_np),
        "SSIM": float(ssim),
        "MS_SSIM_total": float(msssim),
        "VIF": float(vif),
    }


def evaluate_dataset(
    model: EndToEndV8,
    dataset: str,
    data_base: Path,
    output_root: Path,
    device: torch.device,
    progress_every: int,
) -> dict[str, float]:
    ir_subdir, vi_subdir = DATASETS[dataset]
    pairs = paired_images(data_base / dataset, ir_subdir, vi_subdir)
    dataset_root = output_root / dataset
    fused_root = dataset_root / "fused"
    records_path = dataset_root / "per_image.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing[record["name"]] = record
    started = time.time()
    with records_path.open("a", encoding="utf-8") as output:
        for index, (name, ir_path, vi_path) in enumerate(pairs, 1):
            if name in existing:
                continue
            ir = load_luminance(ir_path, device)
            vi = load_luminance(vi_path, device)
            if ir.shape != vi.shape:
                raise ValueError(f"{dataset}/{name}: IR {ir.shape} != VI {vi.shape}")
            height, width = ir.shape[-2:]
            pad_h, pad_w = (-height) % 4, (-width) % 4
            if pad_h or pad_w:
                ir_input = F.pad(ir, (0, pad_w, 0, pad_h), mode="reflect")
                vi_input = F.pad(vi, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                ir_input, vi_input = ir, vi
            with torch.inference_mode():
                fused, stats = model(ir_input, vi_input)
                fused = fused[..., :height, :width].clamp(0, 1)
                values = metrics(ir, vi, fused)
            save_image(fused, fused_root / f"{name}.png")
            record = {
                "dataset": dataset,
                "name": name,
                **values,
                "mix_weights": stats["weights"].cpu().tolist(),
                "mix_gain": float(stats["gain"]),
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            existing[name] = record
            if index == 1 or index % progress_every == 0 or index == len(pairs):
                elapsed = time.time() - started
                print(json.dumps({
                    "event": "progress", "dataset": dataset, "done": len(existing),
                    "total": len(pairs), "elapsed_seconds": elapsed,
                    "images_per_second": max(0, len(existing)) / max(elapsed, 1e-9),
                }), flush=True)
    summary = {key: float(np.mean([row[key] for row in existing.values()])) for key in METRIC_NAMES}
    summary["N"] = len(existing)
    (dataset_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "dataset_complete", "dataset": dataset, **summary}), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = EndToEndV8().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for dataset in args.datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        all_results[dataset] = evaluate_dataset(
            model, dataset, Path(args.data_base), output_root, device, args.progress_every
        )
    (output_root / "all_summary.json").write_text(
        json.dumps(all_results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "all_complete", "results": all_results}), flush=True)


if __name__ == "__main__":
    main()
