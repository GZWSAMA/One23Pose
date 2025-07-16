import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, List
import collections
from torch import Tensor
from itertools import repeat
from models.SpaTrackV2.utils.model_utils import bilinear_sampler
from models.SpaTrackV2.models.blocks import CrossAttnBlock as CrossAttnBlock_F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
# import flash_attn
EPS = 1e-6


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
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
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
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom)
    return mean

class GeometryEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(GeometryEncoder, self).__init__()
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

        self.conv2 = nn.Conv2d(
            output_dim * 5 // 4,
            output_dim,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim, output_dim, kernel_size=1)
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

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        a = self.layer1(x)
        b = self.layer2(a)
        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        x = self.conv2(torch.cat([a, b], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

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

    def forward(self, x):
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

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x

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
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
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
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.inner_dim = inner_dim
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None, flash=True):
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, self.inner_dim // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, self.inner_dim // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, self.inner_dim // h).permute(0, 2, 1, 3)

        if (
            (N1 < 64 and N2 < 64) or
            (B > 1e4) or
            (q.shape[1] != k.shape[1]) or
            (q.shape[1] % k.shape[1] != 0)
        ):
            flash = False


        if flash == False:
            sim = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                sim = sim + attn_bias
            if sim.abs().max() > 1e2:
                import pdb; pdb.set_trace()
            attn = sim.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N1, self.inner_dim)
        else:

            input_args = [x.contiguous() for x in [q, k, v]]
            try:
                # print(f"q.shape: {q.shape}, dtype: {q.dtype}, device: {q.device}")
                # print(f"Flash SDP available: {torch.backends.cuda.flash_sdp_enabled()}")
                # print(f"Flash SDP allowed: {torch.backends.cuda.enable_flash_sdp}")
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    x = F.scaled_dot_product_attention(*input_args).permute(0,2,1,3).reshape(B,N1,-1)  # type: ignore
            except Exception as e:
                print(e)

        if self.to_out.bias.dtype != x.dtype:
            x = x.to(self.to_out.bias.dtype)

        return self.to_out(x)

class CrossAttnBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(context_dim)
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
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

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x

class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = Attention,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
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

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.num_heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
        patch_feat=False,
        patch_dim=128,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        
        if patch_feat==False:
            self.virual_tracks = nn.Parameter(
                torch.randn(1, num_virtual_tracks, 1, hidden_size)
            )
            self.num_virtual_tracks = num_virtual_tracks
        else:
            self.patch_proj = nn.Linear(patch_dim, hidden_size, bias=True)
        
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True, patch_feat=None):
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        if patch_feat is None:
            virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
            tokens = torch.cat([tokens, virtual_tokens], dim=1)
        else:
            patch_feat = self.patch_proj(patch_feat.detach())
            tokens = torch.cat([tokens, patch_feat], dim=1)
            self.num_virtual_tracks = patch_feat.shape[1]

        _, N, _, _ = tokens.shape
        j = 0
        layers = []
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = torch.utils.checkpoint.checkpoint(
                self.time_blocks[i],
                time_tokens
            )

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = torch.utils.checkpoint.checkpoint(
                    self.space_virtual2point_blocks[j],
                    virtual_tokens, point_tokens, mask
                )

                virtual_tokens = torch.utils.checkpoint.checkpoint(
                    self.space_virtual_blocks[j],
                    virtual_tokens
                )
                
                point_tokens = torch.utils.checkpoint.checkpoint(
                    self.space_point2virtual_blocks[j],
                    point_tokens, virtual_tokens, mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow
    
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = alpha * (1 - p_t) ** gamma * ce_loss
    return loss.mean()

def balanced_binary_cross_entropy(logits, targets, balance_weight=1.0, eps=1e-6, reduction="mean", pos_bias=0.0, mask=None):
    """
    logits: Tensor of arbitrary shape
    targets: same shape as logits
    balance_weight: scaling the loss
    reduction: 'mean', 'sum', or 'none'
    """
    targets = targets.float()
    positive = (targets == 1).float().sum()
    total = targets.numel()
    positive_ratio = positive / (total + eps)

    pos_weight = (1 - positive_ratio) / (positive_ratio + eps)
    pos_weight = pos_weight.clamp(min=0.1, max=10.0)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight+pos_bias,
        reduction=reduction
    )
    if mask is not None:
        loss = (loss * mask).sum() / (mask.sum() + eps)
    return balance_weight * loss

def sequence_loss(
    flow_preds,
    flow_gt,
    valids,
    vis=None,
    gamma=0.8,
    add_huber_loss=False,
    loss_only_for_visible=False,
    depth_sample=None,
    z_unc=None,
    mask_traj_gt=None
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i][:,:,:flow_gt[j].shape[2]]
            if flow_pred.shape[-1] == 3:
                flow_pred[...,2] = flow_pred[...,2]
            if add_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)
            else:
                if flow_gt[j][...,2].abs().max() != 0:
                    track_z_loss = (flow_pred- flow_gt[j])[...,2].abs().mean()
                    if mask_traj_gt is not None:
                        track_z_loss = ((flow_pred- flow_gt[j])[...,2].abs() * mask_traj_gt.permute(0,2,1)).sum() / (mask_traj_gt.sum(dim=1)+1e-6)
                else:
                    track_z_loss = 0
                i_loss = (flow_pred[...,:2] - flow_gt[j][...,:2]).abs() # B, S, N, 2
            # print((flow_pred - flow_gt[j])[...,2].abs()[vis[j].bool()].mean())   
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids[j].clone()[:,:, :flow_gt[j].shape[2]]  # Ensure valid_ has the same shape as i_loss
            valid_ = valid_ * (flow_gt[j][...,:2].norm(dim=-1) > 0).float()
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            # print(reduce_masked_mean(i_loss, valid_).item(), track_z_loss.item()/16)
            flow_loss += i_weight * (reduce_masked_mean(i_loss, valid_) + track_z_loss + 10*reduce_masked_mean(i_loss, valid_* vis[j]))
            # if flow_loss > 5e2:
            #     import pdb; pdb.set_trace()
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)

def sequence_loss_xyz(
    flow_preds,
    flow_gt,
    valids,
    intrs,
    vis=None,
    gamma=0.8,
    add_huber_loss=False,
    loss_only_for_visible=False,
    mask_traj_gt=None
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i][:,:,:flow_gt[j].shape[2]]
            flow_gt_ = flow_gt[j]
            flow_gt_one = torch.cat([flow_gt_[...,:2], torch.ones_like(flow_gt_[:,:,:,:1])], dim=-1)
            flow_gt_cam = torch.einsum('btsc,btnc->btns', torch.inverse(intrs), flow_gt_one)
            flow_gt_cam *= flow_gt_[...,2:3].abs()
            flow_gt_cam[...,2] *= torch.sign(flow_gt_cam[...,2])

            if add_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt_cam, delta=6.0)
            else:
                i_loss = (flow_pred- flow_gt_cam).norm(dim=-1,keepdim=True) # B, S, N, 2
                
            # print((flow_pred - flow_gt[j])[...,2].abs()[vis[j].bool()].mean())   
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids[j].clone()[:,:, :flow_gt[j].shape[2]]  # Ensure valid_ has the same shape as i_loss
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            # print(reduce_masked_mean(i_loss, valid_).item(), track_z_loss.item()/16)
            flow_loss += i_weight * (reduce_masked_mean(i_loss, valid_)) * 1000
            # if flow_loss > 5e2:
            #     import pdb; pdb.set_trace()
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)

def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def sequence_BCE_loss(vis_preds, vis_gts, mask=None):
    total_bce_loss = 0.0
    for j in range(len(vis_preds)):
        n_predictions = len(vis_preds[j])
        bce_loss = 0.0
        for i in range(n_predictions):
            N_gt = vis_gts[j].shape[-1]
            if mask is not None:
                vis_loss = balanced_binary_cross_entropy(vis_preds[j][i][...,:N_gt], vis_gts[j], mask=mask[j], reduction="none")
            else:
                vis_loss = balanced_binary_cross_entropy(vis_preds[j][i][...,:N_gt], vis_gts[j]) + focal_loss(vis_preds[j][i][...,:N_gt], vis_gts[j])
            # print(vis_loss, ((torch.sigmoid(vis_preds[j][i][...,:N_gt])>0.5).float() - vis_gts[j]).abs().sum())
            bce_loss += vis_loss
        bce_loss = bce_loss / n_predictions
        total_bce_loss += bce_loss
    return total_bce_loss / len(vis_preds)


def sequence_prob_loss(
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    expected_dist_thresh: float = 12.0,
):
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            N_gt = target_points[j].shape[2]
            err = torch.sum((tracks[j][i].detach()[:,:,:N_gt,:2] - target_points[j][...,:2]) ** 2, dim=-1)
            valid = (err <= expected_dist_thresh**2).float()
            logprob = balanced_binary_cross_entropy(confidence[j][i][...,:N_gt], valid, reduction="none")
            logprob *= visibility[j]
            logprob = torch.mean(logprob, dim=[1, 2])
            logprob_loss += logprob
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)


def sequence_dyn_prob_loss(
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    expected_dist_thresh: float = 6.0,
):
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=-1)
            valid = (err <= expected_dist_thresh**2).float()
            valid = (valid.sum(dim=1) > 0).float() 
            logprob = balanced_binary_cross_entropy(confidence[j][i].mean(dim=1), valid, reduction="none")
            # logprob *= visibility[j]
            logprob = torch.mean(logprob, dim=[0, 1])
            logprob_loss += logprob
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)


def masked_mean(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean


def masked_mean_var(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim), mask_var.squeeze(dim)

class NeighborTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float):
        super().__init__()
        self.dim = dim
        self.output_token_1 = nn.Parameter(torch.randn(1, dim))
        self.output_token_2 = nn.Parameter(torch.randn(1, dim))
        self.xblock1_2 = CrossAttnBlock(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim, mlp_ratio=mlp_ratio)
        self.xblock2_1 = CrossAttnBlock(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim, mlp_ratio=mlp_ratio)
        self.aggr1 = Attention(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim)
        self.aggr2 = Attention(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        from einops import rearrange, repeat
        import torch.utils.checkpoint as checkpoint
        
        assert len (x.shape) == 3, "x should be of shape (B, N, D)"
        assert len (y.shape) == 3, "y should be of shape (B, N, D)"
        
        # not work so well ...

        def forward_chunk(x, y):
            new_x = self.xblock1_2(x, y)
            new_y = self.xblock2_1(y, x)
            out1 = self.aggr1(repeat(self.output_token_1, 'n d -> b n d', b=x.shape[0]), context=new_x)
            out2 = self.aggr2(repeat(self.output_token_2, 'n d -> b n d', b=x.shape[0]), context=new_y)
            return out1 + out2
            
        return checkpoint.checkpoint(forward_chunk, x, y)


class CorrPointformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float):
        super().__init__()
        self.dim = dim
        self.xblock1_2 = CrossAttnBlock(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim, mlp_ratio=mlp_ratio)
        # self.xblock2_1 = CrossAttnBlock(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim, mlp_ratio=mlp_ratio)
        self.aggr = CrossAttnBlock(dim, context_dim=dim, num_heads=num_heads, dim_head=head_dim, mlp_ratio=mlp_ratio)
        self.out_proj = nn.Linear(dim, 2*dim)

    def forward(self, query: torch.Tensor, target: torch.Tensor, target_rel_pos: torch.Tensor) -> torch.Tensor:
        from einops import rearrange, repeat
        import torch.utils.checkpoint as checkpoint

        def forward_chunk(query, target, target_rel_pos):
            new_query = self.xblock1_2(query, target).mean(dim=1, keepdim=True)
            # new_target = self.xblock2_1(target, query).mean(dim=1, keepdim=True)
            # new_aggr = new_query + new_target
            out = self.aggr(new_query, target+target_rel_pos)  # (potential delta xyz)  (target - center)
            out = self.out_proj(out)
            return out
            
        return checkpoint.checkpoint(forward_chunk, query, target, target_rel_pos)