import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int64
from torch import Tensor, nn

from models.SpaTrackV2.models.tracker3D.delta_utils.blocks import (
    Attention,
    AttnBlock,
    BasicEncoder,
    CorrBlock,
    Mlp,
    ResidualBlock,
    Upsample,
    cam2pix,
    pix2cam,
)

from models.SpaTrackV2.models.blocks import bilinear_sampler

def get_grid(height, width, shape=None, dtype="torch", device="cpu", align_corners=True, normalize=True):
    H, W = height, width
    S = shape if shape else []
    if align_corners:
        x = torch.linspace(0, 1, W, device=device)
        y = torch.linspace(0, 1, H, device=device)
        if not normalize:
            x = x * (W - 1)
            y = y * (H - 1)
    else:
        x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
        y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
        if not normalize:
            x = x * W
            y = y * H
    x_view, y_view, exp = [1 for _ in S] + [1, -1], [1 for _ in S] + [-1, 1], S + [H, W]
    x = x.view(*x_view).expand(*exp)
    y = y.view(*y_view).expand(*exp)
    grid = torch.stack([x, y], dim=-1)
    if dtype == "numpy":
        grid = grid.numpy()
    return grid

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

        self.flash = flash

        self.query_proj = nn.Linear(num_heads * query_dim, num_heads * query_dim, bias=qkv_bias)
        self.key_proj = nn.Linear(num_heads * query_dim, num_heads * query_dim, bias=qkv_bias)
        self.value_proj = nn.Linear(num_heads * self.value_size, num_heads * self.value_size, bias=qkv_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=qkv_bias)

        self.scale = 1.0 / math.sqrt(self.query_dim)
        # self.training_length = 24

        # bias_forward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(self.training_length)
        # bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
        # bias_backward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(self.training_length, reverse=True)
        # bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)

        # self.register_buffer("precomputed_attn_bias", torch.cat([bias_forward, bias_backward], dim=0), persistent=False)

    def forward(self, x, context, attn_bias=None):
        B, N1, C = x.size()

        q = self._linear_projection(x, self.query_dim, self.query_proj)  # [T', H, Q=K]
        k = self._linear_projection(context, self.query_dim, self.key_proj)  # [T, H, K]
        v = self._linear_projection(context, self.value_size, self.value_proj)  # [T, H, V]

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

        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        #     attn = F.scaled_dot_product_attention(query_heads, key_heads, value_heads, attn_mask=attn_bias, scale=1.0 / math.sqrt(self.query_dim))
        # else:

        #     sim = (query_heads @ key_heads.transpose(-2, -1)) * self.scale

        #     if attn_bias is not None:
        #         sim = sim + attn_bias
        #     attn = sim.softmax(dim=-1)

        #     attn = (attn @ value_heads)
        # attn = attn.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.final_proj(x)  # [T', D']

    def _linear_projection(self, x, head_size, proj_layer):
        batch_size, sequence_length, _ = x.shape
        y = proj_layer(x)
        y = y.reshape((batch_size, sequence_length, self.num_heads, head_size))

        return y


class UpsampleCrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = RelativeAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, attn_bias=None):
        x = x + self.cross_attn(x=self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderUpsampler(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int = None, stride: int = 4):
        super().__init__()

        self.stride = stride

        if out_channels is None:
            out_channels = middle_channels

        self.conv_in = nn.Conv2d(in_channels, middle_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.norm1 = nn.GroupNorm(num_groups=middle_channels // 8, num_channels=middle_channels, eps=1e-6)

        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i in range(int(math.log2(self.stride))):
            self.res_blocks.append(ResidualBlock(middle_channels, middle_channels))
            self.upsample_blocks.append(Upsample(middle_channels, with_conv=True))

            # in_channels = middle_channels

        self.norm2 = nn.GroupNorm(num_groups=middle_channels // 8, num_channels=middle_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(middle_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.initialize_weight()

    def initialize_weight(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.res_blocks.apply(_basic_init)
        self.conv_in.apply(_basic_init)
        self.conv_out.apply(_basic_init)

    def forward(
        self,
        x: Float[Tensor, "b c1 h_down w_down"],
        mode: str = "nearest",
    ) -> Float[Tensor, "b c1 h_up w_up"]:

        x = F.relu(self.norm1(self.conv_in(x)))

        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            x = self.upsample_blocks[i](x, mode=mode)

        x = self.conv_out(F.relu(self.norm2(x)))
        return x


class UpsampleTransformer(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 4,
        latent_dim: int = 128,
        n_heads: int = 4,
        num_attn_blocks: int = 2,
        use_rel_emb: bool = True,
        flash: bool = False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim

        self.n_heads = n_heads

        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim, middle_channels=self.latent_dim, out_channels=self.latent_dim
        )

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(latent_dim + 64, latent_dim + 64, num_heads=n_heads, mlp_ratio=4, flash=flash)
                for _ in range(num_attn_blocks)
            ]
        )

        self.flow_mlp = nn.Sequential(
            nn.Conv2d(2 * 16, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Linear(latent_dim + 64, kernel_size * kernel_size, bias=True)

        if use_rel_emb:
            self.rpb_attnup = nn.Parameter(torch.zeros(kernel_size * kernel_size))
            torch.nn.init.trunc_normal_(self.rpb_attnup, std=0.1, mean=0.0, a=-2.0, b=2.0)
        else:
            self.rpb_attnup = None

    def forward(
        self,
        feat_map: Float[Tensor, "b c1 h w"],
        flow_map: Float[Tensor, "b c2 h w"],
    ):
        B = feat_map.shape[0]
        H_down, W_down = feat_map.shape[-2:]
        # x0, y0 = x0y0

        feat_map_up = self.attnup_feat_cnn(feat_map)  # learnable upsample by 4
        # feat_map_down = F.interpolate(feat_map_up, scale_factor=1/self.stride, mode='nearest') # B C H*4 W*4
        feat_map_down = feat_map
        # depths_down = F.interpolate(depths, scale_factor=1/self.stride, mode='nearest')

        # NOTE prepare attention bias
        # depths_down_ = torch.stack([depths_down[b, :, y0_:y0_+H_down, x0_:x0_+W_down] for b, (x0_,y0_) in enumerate(zip(x0, y0))], dim=0)
        # depths_ = torch.stack([depths[b, :, y0_*4:y0_*4+H_down*4, x0_*4:x0_*4+W_down*4] for b, (x0_,y0_) in enumerate(zip(x0, y0))], dim=0)
        # guidance_downsample = F.interpolate(guidance, size=(H, W), mode='nearest')
        pad_val = (self.kernel_size - 1) // 2
        # depths_down_padded = F.pad(depths_down_, (pad_val, pad_val, pad_val, pad_val), "replicate")

        if self.rpb_attnup is not None:
            relative_pos_attn_map = self.rpb_attnup.view(1, 1, -1, 1, 1).repeat(
                B, self.n_heads, 1, H_down * 4, W_down * 4
            )
            relative_pos_attn_map = rearrange(relative_pos_attn_map, "b k n h w -> (b h w) k 1 n")
            attn_bias = relative_pos_attn_map
        else:
            attn_bias = None

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val)  # B C*kernel**2 H W
        context = rearrange(context, "b c (h w) -> b c h w", h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=self.stride, mode="nearest")  # B C*kernel**2 H*4 W*4
        context = rearrange(context, "b (c i j) h w -> (b h w) (i j) c", i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, "b c h w -> (b h w) 1 c")

        assert flow_map.shape[-2:] == feat_map.shape[-2:]

        flow_map = rearrange(flow_map, "b t c h w -> b (t c) h w")
        flow_map = self.flow_mlp(flow_map)

        nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val)  # B C*kernel**2 H W
        nn_flow_map = rearrange(nn_flow_map, "b c (h w) -> b c h w", h=H_down, w=W_down)
        nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.stride, mode="nearest")  # B C*kernel**2 H*4 W*4
        nn_flow_map = rearrange(
            nn_flow_map, "b (c i j) h w -> (b h w) (i j) c", i=self.kernel_size, j=self.kernel_size
        )

        up_flow_map = F.interpolate(flow_map, scale_factor=4, mode="nearest")  # NN up # b 2 h w
        up_flow_map = rearrange(up_flow_map, "b c h w -> (b h w) 1 c")

        context = torch.cat([context, nn_flow_map], dim=-1)
        x = torch.cat([x, up_flow_map], dim=-1)

        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context, attn_bias)

        mask_out = self.out(x)
        mask_out = F.softmax(mask_out, dim=-1)
        mask_out = rearrange(mask_out, "(b h w) 1 c -> b c h w", h=H_down * self.stride, w=W_down * self.stride)

        return mask_out


def get_alibi_slope(num_heads):
    x = (24) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)]).float()


class UpsampleTransformerAlibi(nn.Module):
    def __init__(
            self, 
            kernel_size: int = 3, 
            stride: int = 4, 
            latent_dim: int = 128, 
            n_heads: int = 4, 
            num_attn_blocks: int = 2, 
            upsample_factor: int = 4,
        ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.upsample_factor = upsample_factor

        self.n_heads = n_heads

        self.attnup_feat_cnn = DecoderUpsampler(
            in_channels=self.latent_dim,
            middle_channels=self.latent_dim,
            out_channels=self.latent_dim,
            # stride=self.upsample_factor
        )

        self.cross_blocks = nn.ModuleList(
            [
                UpsampleCrossAttnBlock(
                    latent_dim+64, 
                    latent_dim+64, 
                    num_heads=n_heads, 
                    mlp_ratio=4, 
                    flash=False
                )
                for _ in range(num_attn_blocks)
            ]
        )

        self.flow_mlp = nn.Sequential(
            nn.Conv2d(3*32, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Linear(latent_dim+64, kernel_size*kernel_size, bias=True)
        

        alibi_slope = get_alibi_slope(n_heads // 2)
        grid_kernel = get_grid(kernel_size, kernel_size, normalize=False).reshape(kernel_size, kernel_size, 2)
        grid_kernel = grid_kernel - (kernel_size - 1) / 2
        grid_kernel = -torch.abs(grid_kernel)
        alibi_bias = torch.cat([
            alibi_slope.view(-1,1,1) * grid_kernel[..., 0].view(1,kernel_size,kernel_size),
            alibi_slope.view(-1,1,1) * grid_kernel[..., 1].view(1,kernel_size,kernel_size)
        ]) # n_heads, kernel_size, kernel_size

        self.register_buffer("alibi_bias", alibi_bias)


    def forward(
            self, 
            feat_map: Float[Tensor, "b c1 h w"],
            flow_map: Float[Tensor, "b c2 h w"],
        ):
        B = feat_map.shape[0]
        H_down, W_down = feat_map.shape[-2:]

        feat_map_up = self.attnup_feat_cnn(feat_map) # learnable upsample by 4
        if self.upsample_factor != 4:
            additional_scale = float(self.upsample_factor / 4)
            if additional_scale > 1:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='bilinear', align_corners=False)
            else:
                feat_map_up = F.interpolate(feat_map_up, scale_factor=additional_scale, mode='nearest')

        feat_map_down = feat_map

        pad_val = (self.kernel_size - 1) // 2

        attn_bias = self.alibi_bias.view(1,self.n_heads,self.kernel_size**2,1,1).repeat(B,1,1,H_down*self.upsample_factor,W_down*self.upsample_factor)
        attn_bias = rearrange(attn_bias, "b k n h w -> (b h w) k 1 n")

        # NOTE prepare context (low-reso feat)
        context = feat_map_down
        context = F.unfold(context, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        context = rearrange(context, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        context = F.interpolate(context, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        context = rearrange(context, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)

        # NOTE prepare queries (high-reso feat)
        x = feat_map_up
        x = rearrange(x, 'b c h w -> (b h w) 1 c')

        assert flow_map.shape[-2:] == feat_map.shape[-2:]

        flow_map = rearrange(flow_map, 'b t c h w -> b (t c) h w')
        flow_map = self.flow_mlp(flow_map)

        nn_flow_map = F.unfold(flow_map, kernel_size=self.kernel_size, padding=pad_val) # B C*kernel**2 H W
        nn_flow_map = rearrange(nn_flow_map, 'b c (h w) -> b c h w', h=H_down, w=W_down)
        nn_flow_map = F.interpolate(nn_flow_map, scale_factor=self.upsample_factor, mode='nearest') # B C*kernel**2 H*4 W*4
        nn_flow_map = rearrange(nn_flow_map, 'b (c i j) h w -> (b h w) (i j) c', i=self.kernel_size, j=self.kernel_size)
        up_flow_map = F.interpolate(flow_map, scale_factor=self.upsample_factor, mode="nearest") # NN up # b 2 h w
        up_flow_map = rearrange(up_flow_map, 'b c h w -> (b h w) 1 c')
        context = torch.cat([context, nn_flow_map], dim=-1)
        x = torch.cat([x, up_flow_map], dim=-1)
        for lvl in range(len(self.cross_blocks)):
            x = self.cross_blocks[lvl](x, context, attn_bias)

        mask_out = self.out(x)
        mask_out = F.softmax(mask_out, dim=-1)
        mask_out = rearrange(mask_out, '(b h w) 1 c -> b c h w', h=H_down*self.upsample_factor, w=W_down*self.upsample_factor)

        return mask_out