# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
import collections
from functools import partial
from itertools import repeat
import torchvision.models as tvm
from torch.utils.checkpoint import checkpoint
from models.monoD.depth_anything.dpt import DPTHeadEnc, DPTHead
from typing import Union, Tuple
from torch import Tensor

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

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None,
                  num_heads=8, dim_head=48, qkv_bias=False, flash=False):
        super().__init__()
        inner_dim = self.inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads
        self.flash = flash

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x, context=None, attn_bias=None):
        B, N1, _ = x.shape
        C = self.inner_dim
        h = self.heads
        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if self.flash==False:
                sim = (q @ k.transpose(-2, -1)) * self.scale
                if attn_bias is not None:
                    sim = sim + attn_bias
                if sim.abs().max()>1e2:
                    import pdb; pdb.set_trace()
                attn = sim.softmax(dim=-1)
                x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
            else:
                input_args = [x.contiguous() for x in [q, k, v]]
                x = F.scaled_dot_product_attention(*input_args).permute(0,2,1,3).reshape(B,N1,-1)  # type: ignore
            
            if self.to_out.bias.dtype != x.dtype:
                x = x.to(self.to_out.bias.dtype)
            
        return self.to_out(x)


class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast("cuda", enabled=self.amp, dtype = self.amp_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = True, amp_dtype = torch.float16):
        super().__init__()
        # in case the Internet connection is not stable, please load the DINOv2 locally
        self.dinov2_vitl14 = torch.hub.load('models/torchhub/facebookresearch_dinov2_main',
                                          'dinov2_{:}14'.format("vitl"), source='local', pretrained=False)
        
        state_dict = torch.load("models/monoD/zoeDepth/ckpts/dinov2_vitl14_pretrain.pth")
        self.dinov2_vitl14.load_state_dict(state_dict, strict=True)


        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)

        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid

class Dinov2(nn.Module):
    def __init__(self, amp = True, amp_dtype = torch.float16):
        super().__init__()
        # in case the Internet connection is not stable, please load the DINOv2 locally
        self.dinov2_vitl14 = torch.hub.load('models/torchhub/facebookresearch_dinov2_main',
                                          'dinov2_{:}14'.format("vitl"), source='local', pretrained=False)
        
        state_dict = torch.load("models/monoD/zoeDepth/ckpts/dinov2_vitl14_pretrain.pth")
        self.dinov2_vitl14.load_state_dict(state_dict, strict=True)

        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            self.dinov2_vitl14 = self.dinov2_vitl14.to(self.amp_dtype)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        mean_ = torch.tensor([0.485, 0.456, 0.406], 
                             device=x.device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225],
                            device=x.device).view(1, 3, 1, 1)
        x = (x+1)/2
        x = (x - mean_)/std_
        h_re, w_re = 560, 560
        x_resize = F.interpolate(x, size=(h_re, w_re),
                                  mode='bilinear', align_corners=True)
        if not upsample:
            with torch.no_grad():
                dinov2_features_16 = self.dinov2_vitl14.forward_features(x_resize.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,h_re//14, w_re//14)
                del dinov2_features_16
        features_16 = F.interpolate(features_16, size=(H//8, W//8), mode="bilinear", align_corners=True)
        return features_16

class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                  flash=False, ckpt_fwd=False, debug=False, **block_kwargs):
        super().__init__()
        self.debug=debug
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.flash=flash

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, flash=flash, 
            **block_kwargs
        )        
        self.ls = LayerScale(hidden_size, init_values=0.005)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
        )
        self.ckpt_fwd = ckpt_fwd
    def forward(self, x):
        if self.debug:
            print(x.max(), x.min(), x.mean())
        if self.ckpt_fwd:
            x = x + checkpoint(self.attn, self.norm1(x), use_reentrant=False)
        else:
            x = x + self.attn(self.norm1(x))

        x = x + self.ls(self.mlp(self.norm2(x)))
        return x
    
class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, head_dim=48,
                 flash=False, ckpt_fwd=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)

        self.cross_attn = Attention(
            hidden_size, context_dim=context_dim, dim_head=head_dim,
            num_heads=num_heads, qkv_bias=True, **block_kwargs, flash=flash,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.ckpt_fwd = ckpt_fwd
    def forward(self, x, context):
        if self.ckpt_fwd:
            with autocast():
                x = x + checkpoint(self.cross_attn,
                                    self.norm1(x), self.norm_context(context), use_reentrant=False)
        else:
            with autocast():
                x = x + self.cross_attn(
                    self.norm1(x), self.norm_context(context)
                )
        x = x + self.mlp(self.norm2(x))
        return x


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode=mode)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4, depths_dnG=None):
        B, S, C, H_prev, W_prev = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H_prev, W_prev

        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.depth_pyramid = []
        self.fmaps_pyramid.append(fmaps)
        if depths_dnG is not None:
           self.depth_pyramid.append(depths_dnG) 
        for i in range(self.num_levels - 1):
            if depths_dnG is not None:
                depths_dnG_ = depths_dnG.reshape(B * S, 1, H_prev, W_prev)
                depths_dnG_ = F.avg_pool2d(depths_dnG_, 2, stride=2)
                _, _, H, W = depths_dnG_.shape
                depths_dnG = depths_dnG_.reshape(B, S, 1, H, W)
                self.depth_pyramid.append(depths_dnG)
            fmaps_ = fmaps.reshape(B * S, C, H_prev, W_prev)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            H_prev = H
            W_prev = W
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            _, _, _, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corrs = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), coords_lvl)
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)
    
    def corr_sample(self, targets, coords, coords_dp=None):
        B, S, N, C = targets.shape
        r = self.radius
        Dim_c = (2*r+1)**2
        assert C == self.C
        assert S == self.S

        out_pyramid = []
        out_pyramid_dp = []
        for i in range(self.num_levels): 
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )
            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            fmaps = self.fmaps_pyramid[i]
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B*S, C, H, W)
            if len(self.depth_pyramid)>0:
                depths_dnG_i = self.depth_pyramid[i]
                depths_dnG_i = depths_dnG_i.view(B*S, 1, H, W)
                dnG_sample = bilinear_sampler(depths_dnG_i, coords_lvl.view(B*S,1,N*Dim_c,2))
                dp_corrs = (dnG_sample.view(B*S,N,-1) - coords_dp[0]).abs()/coords_dp[0]
                out_pyramid_dp.append(dp_corrs)
            fmap2s_sample = bilinear_sampler(fmap2s, coords_lvl.view(B*S,1,N*Dim_c,2))
            fmap2s_sample = fmap2s_sample.permute(0, 3, 1, 2) # B*S, N*Dim_c, C, -1
            corrs = torch.matmul(targets.reshape(B*S*N, 1, -1), fmap2s_sample.reshape(B*S*N, Dim_c, -1).permute(0, 2, 1))
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        if len(self.depth_pyramid)>0:
            out_dp = torch.cat(out_pyramid_dp, dim=-1)
            self.fcorrD = out_dp.contiguous().float()
        else:
            self.fcorrD = torch.zeros_like(out).contiguous().float()
        return out.contiguous().float()


class EUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=12,
        time_depth=12,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        vq_depth=3,
        add_space_attn=True,
        add_time_attn=True,
        flash=True
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flash = flash
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_size, output_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim, bias=True)
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=50,
            num_blocks=7,
            embedding_dim=384,
            slstm_at=[1],

        )
        self.xlstm_fwd = xLSTMBlockStack(cfg)
        self.xlstm_bwd = xLSTMBlockStack(cfg)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, 
                input_tensor,
                track_mask=None):
        """ Updating with Transformer

        Args:
            input_tensor: B, N, T, C
            arap_embed: B, N, T, C
        """
        B, N, T, C = input_tensor.shape
        x = self.input_transform(input_tensor)

        track_mask = track_mask.permute(0,2,1,3).float()
        fwd_x = x*track_mask
        bwd_x = x.flip(2)*track_mask.flip(2)
        feat_fwd = self.xlstm_fwd(self.norm(fwd_x.view(B*N, T, -1)))
        feat_bwd = self.xlstm_bwd(self.norm(bwd_x.view(B*N, T, -1)))
        feat = (feat_bwd.flip(1) + feat_fwd).view(B, N, T, -1)

        flow = self.flow_head(feat)

        return flow[..., :2], flow[..., 2:]

