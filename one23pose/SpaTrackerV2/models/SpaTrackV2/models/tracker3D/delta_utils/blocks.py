# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
from functools import partial
from itertools import repeat
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SpaTrackV2.models.blocks import bilinear_sampler
from einops import rearrange
from torch import Tensor, einsum


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        zero_init=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        if zero_init:
            self.zero_init()

    def zero_init(self):
        nn.init.constant_(self.fc2.weight, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mode="nearest"):
        x = F.interpolate(x, scale_factor=2.0, mode=mode)
        if self.with_conv:
            x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode="zeros")
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, return_intermediate=False):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        # a = _bilinear_intepolate(a)
        # b = _bilinear_intepolate(b)
        # c = _bilinear_intepolate(c)
        # d = _bilinear_intepolate(d)

        cat_feat = torch.cat(
            [_bilinear_intepolate(a), _bilinear_intepolate(b), _bilinear_intepolate(c), _bilinear_intepolate(d)], dim=1
        )
        x = self.conv2(cat_feat)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        # breakpoint()
        if return_intermediate:
            if self.stride == 4:
                return x, a, c  # 128, h/4, w/4, - 64, h/2, w/2 - 128, h/8, w/8
            elif self.stride == 8:
                return x, b, d
            else:
                raise NotImplementedError
        return x


class CorrBlockFP16:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        multiple_track_feats=False,
        padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # breakpoint()
            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        del self.corrs_pyramid

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            # breakpoint()
            self.corrs_pyramid.append(corrs)


class CorrBlock:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        multiple_track_feats=False,
        padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords, delete=True):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            # breakpoint()

            # t1 = time.time()
            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            # t2 = time.time()

            # print(coords_lvl.shape, t2 - t1)
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        if delete:
            del self.corrs_pyramid

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            # breakpoint()
            self.corrs_pyramid.append(corrs)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        num_heads=8,
        dim_head=48,
        qkv_bias=False,
        flash=False,
        alibi=False,
        zero_init=False,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.flash = flash
        self.alibi = alibi

        if zero_init:
            self.zero_init()
            # if self.alibi:
            #     self.training_length = 24

            #     bias_forward = get_alibi_slope(self.heads // 2) * get_relative_positions(self.training_length)
            #     bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
            #     bias_backward = get_alibi_slope(self.heads // 2) * get_relative_positions(self.training_length, reverse=True)
            #     bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)

            #     self.precomputed_attn_bias = self.register_buffer("precomputed_attn_bias", torch.cat([bias_forward, bias_backward], dim=0), persistent=False)

    def zero_init(self):
        nn.init.constant_(self.to_out.weight, 0)
        nn.init.constant_(self.to_out.bias, 0)

        # breakpoint()

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h)
        context = default(context, x)
        N2 = context.shape[1]
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.reshape(B, N2, h, C // h)
        v = v.reshape(B, N2, h, C // h)

        if self.flash:
            with torch.autocast(device_type="cuda", enabled=True):
                x = flash_attn_func(q.half(), k.half(), v.half())
                x = x.reshape(B, N1, C)
            x = x.float()
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            sim = (q @ k.transpose(-2, -1)) * self.scale

            if attn_bias is not None:
                sim = sim + attn_bias
            attn = sim.softmax(dim=-1)

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N1, C)
        x = self.to_out(x)
        return x

    def forward_noattn(self, x):
        # B, N1, C = x.shape
        # h = self.heads
        _, x = self.to_kv(x).chunk(2, dim=-1)
        # x = x.reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        # x = x.transpose(1, 2).reshape(B, N1, C)
        x = self.to_out(x)

        return x


def get_relative_positions(seq_len, reverse=False, device="cpu"):
    x = torch.arange(seq_len, device=device)[None, :]
    y = torch.arange(seq_len, device=device)[:, None]
    return torch.tril(x - y) if not reverse else torch.triu(y - x)


def get_alibi_slope(num_heads, device="cpu"):
    x = (24) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device=device, dtype=torch.float32).view(
        -1, 1, 1
    )


class RelativeAttention(nn.Module):
    """Multi-headed attention (MHA) module."""

    def __init__(self, query_dim, num_heads=8, qkv_bias=True, model_size=None, flash=False):
        super(RelativeAttention, self).__init__()

        query_dim = query_dim // num_heads
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.value_size = query_dim
        self.model_size = query_dim * num_heads

        self.qkv_bias = qkv_bias

        self.query_proj = nn.Linear(num_heads * query_dim, num_heads * query_dim, bias=qkv_bias)
        self.key_proj = nn.Linear(num_heads * query_dim, num_heads * query_dim, bias=qkv_bias)
        self.value_proj = nn.Linear(num_heads * self.value_size, num_heads * self.value_size, bias=qkv_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=qkv_bias)

        self.training_length = 24

        bias_forward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(self.training_length)
        bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
        bias_backward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(
            self.training_length, reverse=True
        )
        bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)

        self.register_buffer(
            "precomputed_attn_bias", torch.cat([bias_forward, bias_backward], dim=0), persistent=False
        )

    def forward(self, x, attn_bias=None):
        batch_size, sequence_length, _ = x.size()

        query_heads = self._linear_projection(x, self.query_dim, self.query_proj)  # [T', H, Q=K]
        key_heads = self._linear_projection(x, self.query_dim, self.key_proj)  # [T, H, K]
        value_heads = self._linear_projection(x, self.value_size, self.value_proj)  # [T, H, V]

        if self.training_length == sequence_length:
            new_attn_bias = self.precomputed_attn_bias
        else:
            device = x.device
            bias_forward = get_alibi_slope(self.num_heads // 2, device=device) * get_relative_positions(
                sequence_length, device=device
            )
            bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
            bias_backward = get_alibi_slope(self.num_heads // 2, device=device) * get_relative_positions(
                sequence_length, reverse=True, device=device
            )
            bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)
            new_attn_bias = torch.cat([bias_forward, bias_backward], dim=0)

        if attn_bias is not None:
            attn_bias = attn_bias + new_attn_bias
        else:
            attn_bias = new_attn_bias

        attn = F.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, attn_mask=new_attn_bias, scale=1 / np.sqrt(self.query_dim)
        )
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.final_proj(attn)  # [T', D']

        # attn_logits = torch.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        # attn_logits = attn_logits / np.sqrt(self.query_dim) + new_attn_bias

        # # breakpoint()
        # if attn_bias is not None:
        #     if attn_bias.ndim != attn_logits.ndim:
        #         raise ValueError(f"Mask dimensionality {attn_bias.ndim} must match logits dimensionality {attn_logits.ndim}.")
        #     attn_logits = torch.where(attn_bias, attn_logits, torch.tensor(-1e30))

        # attn_weights = F.softmax(attn_logits, dim=-1)  # [H, T', T]

        # attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        # attn = attn.reshape(batch_size, sequence_length, -1)  # [T', H*V]

        # return self.final_proj(attn)  # [T', D']

    # def _linear_projection(self, x, head_size, proj_layer):
    #     y = proj_layer(x)
    #     *leading_dims, _ = x.shape
    #     return y.reshape((*leading_dims, self.num_heads, head_size))

    def _linear_projection(self, x, head_size, proj_layer):
        y = proj_layer(x)
        batch_size, sequence_length, _ = x.shape
        return y.reshape((batch_size, sequence_length, self.num_heads, head_size)).permute(0, 2, 1, 3)


class AttnBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, attn_class: Callable[..., nn.Module] = Attention, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (mask[:, None] * mask[:, :, None]).unsqueeze(1).expand(-1, self.attn.heads, -1, -1)
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_noattn(self, x):
        x = x + self.attn.forward_noattn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def pix2cam(coords, intr, detach=True):
    """
    Args:
        coords: [B, T, N, 3]
        intr: [B, T, 3, 3]
    """
    if detach:
        coords = coords.detach()

    (
        B,
        S,
        N,
        _,
    ) = coords.shape
    xy_src = coords.reshape(B * S * N, 3)
    intr = intr[:, :, None, ...].repeat(1, 1, N, 1, 1).reshape(B * S * N, 3, 3)
    xy_src = torch.cat([xy_src[..., :2], torch.ones_like(xy_src[..., :1])], dim=-1)
    xyz_src = (torch.inverse(intr) @ xy_src[..., None])[..., 0]
    dp_pred = coords[..., 2]
    xyz_src_ = xyz_src * (dp_pred.reshape(S * N, 1))
    xyz_src_ = xyz_src_.reshape(B, S, N, 3)
    return xyz_src_


def cam2pix(coords, intr):
    """
    Args:
        coords: [B, T, N, 3]
        intr: [B, T, 3, 3]
    """
    coords = coords.detach()
    (
        B,
        S,
        N,
        _,
    ) = coords.shape
    xy_src = coords.reshape(B * S * N, 3).clone()
    intr = intr[:, :, None, ...].repeat(1, 1, N, 1, 1).reshape(B * S * N, 3, 3)
    xy_src = xy_src / (xy_src[..., 2:] + 1e-5)
    xyz_src = (intr @ xy_src[..., None])[..., 0]
    dp_pred = coords[..., 2]
    xyz_src[..., 2] *= dp_pred.reshape(S * N)
    xyz_src = xyz_src.reshape(B, S, N, 3)
    return xyz_src


class BroadMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(BroadMultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = rearrange(Q.squeeze(), "i (heads d) -> heads i d", heads=self.heads)
        K = rearrange(K, "b j (heads d) -> b heads j d", heads=self.heads)

        dots = einsum("hid, bhjd -> bhij", Q, K) * self.scale  # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, _, _ = K.shape
        _, N, _ = Q.shape

        V = rearrange(V, "b j (heads d) -> b heads j d", heads=self.heads)

        out = einsum("bhij, bhjd -> bhid", attn, V)
        out = rearrange(out, "b heads n d -> b n (heads d)", b=B, n=N)

        return out


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        qk_dim,
        v_dim,
        query_token_dim,
        tgt_token_dim,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        dropout=0.0,
    ):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim**-0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = (
            nn.Linear(query_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, v_dim, bias=True),
        )

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, tgt_token):
        """
        x: [BH1W1, H3W3, D]
        """
        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)

        x = self.multi_head_attn(q, k, v)

        x = short_cut + self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return rearrange(x, "b h w c -> b c h w")


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
        x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
        min_deg: int, the minimum (inclusive) degree of the encoding.
        max_deg: int, the maximum (exclusive) degree of the encoding.
        legacy_posenc_order: bool, keep the same ordering as the original tf code.

    Returns:
        encoded: torch.Tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device)
    if legacy_posenc_order:
        xb = x[..., None, :] * scales[:, None]
        four_feat = torch.reshape(torch.sin(torch.stack([xb, xb + 0.5 * np.pi], dim=-2)), list(x.shape[:-1]) + [-1])
    else:
        xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def gaussian2D2(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h / h.sum()
