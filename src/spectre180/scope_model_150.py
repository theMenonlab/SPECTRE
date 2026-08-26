"""Calibration-conditioned SPECTRE backbone derived from NAFNet.

NAFNet portions are used under the upstream MIT license. See
``licenses/NAFNet_LICENSE.txt`` and ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_OUTPUT_CHANNELS = 150
NUM_POLARIZATIONS = 5
NUM_WAVELENGTHS = 30
SIGMA_MIN = 1e-3


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, c, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, c, _, _ = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=(0, 2, 3)), grad_output.sum(dim=(0, 2, 3)), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CalibrationAdapter(nn.Module):
    def __init__(self, mask_channels: int, width: int, num_levels: int):
        super().__init__()
        self.intro = nn.Sequential(
            nn.Conv2d(mask_channels, width, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=True),
        )
        self.downs = nn.ModuleList()
        chan = width
        for _ in range(num_levels):
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2, bias=True))
            chan *= 2

    def forward(self, mask: torch.Tensor) -> list[torch.Tensor]:
        x = self.intro(mask)
        feats = [x]
        for down in self.downs:
            x = down(x)
            feats.append(x)
        return feats


class ScopeBlock(nn.Module):
    def __init__(self, c: int, mask_c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.mask_gate1 = nn.Sequential(nn.Conv2d(mask_c, dw_channel // 2, 1, bias=True), nn.Sigmoid())
        self.mask_gate2 = nn.Sequential(nn.Conv2d(mask_c, c, 1, bias=True), nn.Sigmoid())

    def forward(self, inp: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x) * self.mask_gate1(mask_feat)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = x * self.mask_gate2(mask_feat)
        return y + x * self.gamma


class FactorizedDualOutputHead(nn.Module):
    def __init__(self, width: int, out_channels: int, rank: int = 64):
        super().__init__()
        half_out = out_channels // 2
        self.direct = nn.Conv2d(width, out_channels, 3, padding=1, bias=True)
        self.norm = LayerNorm2d(width)
        self.latents = nn.Sequential(
            nn.Conv2d(width, rank, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(rank, rank, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(rank, rank, 3, padding=1, groups=max(1, rank // 8), bias=True),
            nn.GELU(),
        )
        self.mu_proj = nn.Conv2d(rank, half_out, 1, bias=True)
        self.sigma_proj = nn.Conv2d(rank, half_out, 1, bias=True)
        nn.init.zeros_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.sigma_proj.weight)
        nn.init.zeros_(self.sigma_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.direct(x)
        latent = self.latents(self.norm(x))
        refined = torch.cat([self.mu_proj(latent), self.sigma_proj(latent)], dim=1)
        return base + refined


class ScopeNet150(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = NUM_OUTPUT_CHANNELS * 2,
        mask_channels: int = NUM_OUTPUT_CHANNELS,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: tuple[int, ...] = (2, 2, 4, 8),
        dec_blk_nums: tuple[int, ...] = (2, 2, 2, 2),
        output_head_rank: int = 64,
    ):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)
        self.ending = FactorizedDualOutputHead(width, out_channels, rank=output_head_rank)
        self.cal_adapter = CalibrationAdapter(mask_channels=mask_channels, width=width, num_levels=len(enc_blk_nums))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.ModuleList([ScopeBlock(chan, chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, stride=2))
            chan *= 2

        self.middle_blks = nn.ModuleList([ScopeBlock(chan, chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.ModuleList([ScopeBlock(chan, chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        inp = self.check_image_size(inp)
        mask = self.check_image_size(mask)
        mask_feats = self.cal_adapter(mask)

        x = self.intro(inp)
        encs = []
        for level, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            mask_feat = mask_feats[level]
            for block in encoder:
                x = block(x, mask_feat)
            encs.append(x)
            x = down(x)

        middle_mask = mask_feats[-1]
        for block in self.middle_blks:
            x = block(x, middle_mask)

        for level, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            mask_feat = mask_feats[::-1][level + 1]
            for block in decoder:
                x = block(x, mask_feat)

        x = self.ending(x)
        return x[:, :, :h, :w]


def split_prediction(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = prediction[:, :NUM_OUTPUT_CHANNELS, :, :]
    sigma_raw = prediction[:, NUM_OUTPUT_CHANNELS:, :, :]
    sigma = F.softplus(sigma_raw) + SIGMA_MIN
    return mu, sigma


def build_scope_model_150() -> ScopeNet150:
    return ScopeNet150(width=32, output_head_rank=64)


def expand_120_to_150_channels(tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
    tensor = tensor.transpose(channel_dim, 0)
    original_shape = tensor.shape
    tensor = tensor.reshape(120, -1)
    expanded = torch.zeros((150, tensor.shape[1]), dtype=tensor.dtype)
    src = tensor.reshape(NUM_WAVELENGTHS, 4, -1)
    dst = expanded.reshape(NUM_WAVELENGTHS, NUM_POLARIZATIONS, -1)
    dst[:, :4, :] = src
    dst[:, 4, :] = src[:, 3, :]
    expanded = expanded.reshape((150,) + original_shape[1:])
    return expanded.transpose(0, channel_dim).contiguous()


def warmstart_from_120_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    target_state = model.state_dict()
    loaded = []
    skipped = []

    for key, target_value in target_state.items():
        if key not in source_state:
            skipped.append(key)
            continue
        source_value = source_state[key]
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            loaded.append(key)
            continue

        expanded = None
        if source_value.shape[0] == 120 and target_value.shape[0] == 150:
            expanded = expand_120_to_150_channels(source_value, 0)
        elif source_value.ndim > 1 and source_value.shape[1] == 120 and target_value.shape[1] == 150:
            expanded = expand_120_to_150_channels(source_value, 1)

        if expanded is not None and expanded.shape == target_value.shape:
            target_state[key] = expanded
            loaded.append(key)
        else:
            skipped.append(key)

    model.load_state_dict(target_state, strict=True)
    return {
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "checkpoint_path": str(checkpoint_path),
    }
