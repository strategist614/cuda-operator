"""V10: V7 dual encoders with 3x3 hierarchical manifold routing."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CODE_ROOT = Path("/data/code")
if CODE_ROOT.is_dir() and str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from aecfusion_lite_comprec_stronger_experts import (  # noqa: E402
    GatedDConvBlock,
    LiteDWConvBlock,
    LiteGatedFusionBlock,
    LiteSEEncoder,
)
from region_hierarchical_causal_geometry import (  # noqa: E402
    GeometryOperatorLibrary,
    PatchMeta,
    RHCGConfig,
)


GRID_SIZE = 3
ROUTE_NAMES = ("E", "SPD", "Hyp", "Gr")


def grid_patchify(x: Tensor, grid_size: int = GRID_SIZE) -> tuple[Tensor, PatchMeta]:
    b, c, h, w = x.shape
    ph, pw = math.ceil(h / grid_size), math.ceil(w / grid_size)
    padded = F.pad(x, (0, pw * grid_size - w, 0, ph * grid_size - h), mode="replicate")
    regions = padded.reshape(b, c, grid_size, ph, grid_size, pw)
    regions = regions.permute(0, 2, 4, 3, 5, 1).reshape(b, grid_size * grid_size, ph * pw, c)
    meta = PatchMeta(
        original_hw=(h, w),
        padded_hw=(ph * grid_size, pw * grid_size),
        grid_hw=(grid_size, grid_size),
        pad_hw=(ph * grid_size - h, pw * grid_size - w),
        patch_size=-1,
    )
    meta.region_hw = (ph, pw)
    return regions.contiguous(), meta


def grid_unpatchify(regions: Tensor, meta: PatchMeta) -> Tensor:
    b, l, n, c = regions.shape
    gh, gw = meta.grid_hw
    ph, pw = meta.region_hw
    if (l, n) != (gh * gw, ph * pw):
        raise ValueError("Region tensor does not match grid metadata")
    x = regions.reshape(b, gh, gw, ph, pw, c).permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(b, c, gh * ph, gw * pw)
    h, w = meta.original_hw
    return x[..., :h, :w]


class TwoStageRegionRouter(nn.Module):
    """Select Euclidean vs manifold, then SPD/Hyperbolic/Grassmann."""

    def __init__(self, channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(channels * 3 + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.space_head = nn.Linear(hidden, 2)
        self.manifold_head = nn.Linear(hidden, 3)

    @staticmethod
    def descriptor(regions: Tensor) -> Tensor:
        b, l, _, _ = regions.shape
        mean = regions.mean(dim=2)
        std = regions.std(dim=2, unbiased=False)
        max_abs = regions.abs().amax(dim=2)
        side = int(math.isqrt(l))
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, side, device=regions.device, dtype=regions.dtype),
                torch.linspace(-1, 1, side, device=regions.device, dtype=regions.dtype),
                indexing="ij",
            ), dim=-1,
        ).reshape(1, l, 2).expand(b, -1, -1)
        return torch.cat((mean, std, max_abs, coords), dim=-1)

    def forward(self, regions: Tensor, temperature: float = 1.0, hard: bool = True) -> dict[str, Tensor]:
        hidden = self.backbone(self.descriptor(regions))
        space_logits = self.space_head(hidden)
        manifold_logits = self.manifold_head(hidden)
        if hard:
            if self.training:
                space = F.gumbel_softmax(space_logits, tau=max(temperature, 1e-4), hard=True, dim=-1)
                manifold = F.gumbel_softmax(manifold_logits, tau=max(temperature, 1e-4), hard=True, dim=-1)
            else:
                space = F.one_hot(space_logits.argmax(-1), 2).to(regions.dtype)
                manifold = F.one_hot(manifold_logits.argmax(-1), 3).to(regions.dtype)
        else:
            space = torch.softmax(space_logits / max(temperature, 1e-4), dim=-1)
            manifold = torch.softmax(manifold_logits / max(temperature, 1e-4), dim=-1)
        weights = torch.cat((space[..., :1], space[..., 1:] * manifold), dim=-1)
        return {
            "weights": weights,
            "route_ids": weights.detach().argmax(-1),
            "space_logits": space_logits,
            "manifold_logits": manifold_logits,
        }


class V10Decoder(nn.Module):
    """V7-compatible lightweight U-Net decoder without prior-expert branches."""

    def __init__(self) -> None:
        super().__init__()
        self.proj3 = nn.Sequential(nn.Conv2d(96, 48, 1), nn.GELU())
        self.refine2 = LiteDWConvBlock(96, 48, hidden_ch=48)
        self.proj2 = nn.Sequential(nn.Conv2d(48, 24, 1), nn.GELU())
        self.refine1 = LiteDWConvBlock(48, 24, hidden_ch=24)
        self.out = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1, groups=24), nn.GELU(),
            nn.Conv2d(24, 1, 1), nn.Sigmoid(),
        )

    def forward(self, f3: Tensor, f2: Tensor, f1: Tensor) -> Tensor:
        x = F.interpolate(self.proj3(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine2(torch.cat((x, f2), dim=1))
        x = F.interpolate(self.proj2(x), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.refine1(torch.cat((x, f1), dim=1))
        return self.out(x)


class V10Model(nn.Module):
    """End-to-end V10 model initialized only from V7's dual encoders."""

    def __init__(self, encoder_checkpoint: str | Path, *, router_hidden: int = 128,
                 router_temperature: float = 1.0, geometry_config: dict | None = None) -> None:
        super().__init__()
        self.ir_encoder = LiteSEEncoder(in_ch=1, channels=(24, 48, 96))
        self.vi_encoder = LiteSEEncoder(in_ch=1, channels=(24, 48, 96))
        self._load_encoder_weights(encoder_checkpoint)
        self.fuse_stage1 = nn.Sequential(nn.Conv2d(48, 24, 1), nn.GELU(), GatedDConvBlock(24))
        self.fuse_stage2 = LiteGatedFusionBlock(48)
        self.fuse_stage3 = LiteGatedFusionBlock(96)
        values = {
            "feature_channels": 96, "patch_size": 8, "spd_rank": 24,
            "hyperbolic_dim": 32, "grassmann_dim": 32, "grassmann_rank": 8,
            "curvature": 1.0, "covariance_epsilon": 1e-4,
        }
        if geometry_config:
            values.update(geometry_config)
        self.geometry_config = RHCGConfig(**values)
        self.geometry = GeometryOperatorLibrary(self.geometry_config)
        self.region_router = TwoStageRegionRouter(96, router_hidden)
        self.router_temperature = router_temperature
        self.decoder = V10Decoder()

    def _load_encoder_weights(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        own = self.state_dict()
        loaded = 0
        for prefix in ("ir_encoder.", "vi_encoder."):
            for key, value in state.items():
                if key.startswith(prefix) and key in own and own[key].shape == value.shape:
                    own[key].copy_(value)
                    loaded += 1
        if loaded == 0:
            raise RuntimeError(f"No dual-encoder weights found in {checkpoint}")

    def encode(self, infrared: Tensor, visible: Tensor) -> tuple[Tensor, PatchMeta, tuple[Tensor, Tensor]]:
        ir1, ir2, ir3 = self.ir_encoder(infrared)
        vi1, vi2, vi3 = self.vi_encoder(visible)
        f1 = self.fuse_stage1(torch.cat((ir1, vi1), dim=1))
        f2 = self.fuse_stage2(ir2, vi2)
        f3 = self.fuse_stage3(ir3, vi3)
        regions, meta = grid_patchify(f3, GRID_SIZE)
        return regions, meta, (f2, f1)

    def _route(self, regions: Tensor, hard: bool = True) -> tuple[Tensor, dict[str, Tensor]]:
        outcomes = self.geometry.apply_all(regions)
        routing = self.region_router(regions, self.router_temperature, hard=hard)
        stacked = torch.stack([outcomes[i] for i in range(4)], dim=2)
        routed = (routing["weights"][..., None, None] * stacked).sum(dim=2)
        routing["route_fractions"] = torch.stack([(routing["route_ids"] == i).float().mean() for i in range(4)])
        routing["grid_hw"] = (GRID_SIZE, GRID_SIZE)
        return routed, routing

    def routed_forward(self, infrared: Tensor, visible: Tensor, *, return_stats: bool = False,
                       hard: bool = True):
        regions, meta, (f2, f1) = self.encode(infrared, visible)
        routed, stats = self._route(regions, hard=hard)
        prediction = self.decoder(grid_unpatchify(routed, meta), f2, f1)
        return (prediction, stats) if return_stats else prediction

    def euclidean_forward(self, infrared: Tensor, visible: Tensor) -> Tensor:
        regions, meta, (f2, f1) = self.encode(infrared, visible)
        return self.decoder(grid_unpatchify(regions, meta), f2, f1)

    def trainable_adapter_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def routing_regularization(self, stats: dict[str, Tensor]) -> Tensor:
        sp = torch.softmax(stats["space_logits"], -1).mean((0, 1))
        mp = torch.softmax(stats["manifold_logits"], -1).mean((0, 1))
        return F.kl_div(sp.clamp_min(1e-8).log(), sp.new_tensor([0.5, 0.5]), reduction="sum") + F.kl_div(
            mp.clamp_min(1e-8).log(), mp.new_full((3,), 1 / 3), reduction="sum")
