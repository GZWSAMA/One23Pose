import os, sys
import torch
import torch.amp
import torch.nn.functional as F
import torch.nn as nn
from models.SpaTrackV2.models.tracker3D.co_tracker.utils import (
    EfficientUpdateFormer, AttnBlock, Attention, CrossAttnBlock,
    sequence_BCE_loss, sequence_loss, sequence_prob_loss, sequence_dyn_prob_loss
)
import math
from models.SpaTrackV2.models.tracker3D.co_tracker.utils import (
    Mlp, BasicEncoder, EfficientUpdateFormer, GeometryEncoder, NeighborTransformer
)
import numpy as np
from models.SpaTrackV2.models.tracker3D.spatrack_modules.simple_vit_1d import Transformer,posemb_sincos_1d
from einops import rearrange

def self_grid_pos_embedding(B, T, H, W, level=None):
    import pdb; pdb.set_trace()

def random_se3_transformation(
    batch_size: int = 1,
    max_rotation_angle: float = math.pi,  
    max_translation: float = 1.0,         
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    随机生成刚体变换矩阵（SE(3) Transformation Matrix）。

    Args:
        batch_size (int): 批大小，默认为 1。
        max_rotation_angle (float): 最大旋转角度（弧度），默认 π（180°）。
        max_translation (float): 最大平移量，默认 1.0。
        device (str): 设备（'cpu' 或 'cuda'）。
        dtype (torch.dtype): 数据类型（推荐 float32）。

    Returns:
        torch.Tensor: 形状为 (batch_size, 4, 4) 的齐次变换矩阵。
    """
    # 随机生成旋转矩阵 R (batch_size, 3, 3)
    # 方法 1：使用轴角表示（Axis-Angle）转换为旋转矩阵
    axis = torch.randn(batch_size, 3, device=device, dtype=dtype)  # 随机旋转轴
    axis = axis / torch.norm(axis, dim=1, keepdim=True)  # 归一化
    angle = torch.rand(batch_size, 1, device=device, dtype=dtype) * max_rotation_angle  # 随机角度 [0, max_angle]
    
    # 计算旋转矩阵（Rodrigues' rotation formula）
    K = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    R = I + torch.sin(angle).unsqueeze(-1) * K + (1 - torch.cos(angle).unsqueeze(-1)) * (K @ K)
    
    # 随机生成平移向量 t (batch_size, 3)
    t = (torch.rand(batch_size, 3, device=device, dtype=dtype) - 0.5) * 2 * max_translation
    
    # 组合成齐次变换矩阵 T (batch_size, 4, 4)
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    
    return T

def weighted_procrustes_torch(X, Y, W=None, RT=None):
    """
    Weighted Procrustes Analysis in PyTorch (batched).
    
    Args:
        X: (B, 1, N, 3), source point cloud.
        Y: (B, T, N, 3), target point cloud.
        W: (B, T, N) or (B, 1, N), optional weights for each point.
    
    Returns:
        t: (B, T, 3), optimal translation vectors.
        R: (B, T, 3, 3), optimal rotation matrices.
    """
    device = X.device
    B, T, N, _ = Y.shape
    
    # Default weights: uniform
    if W is None:
        W = torch.ones(B, 1, N, device=device)
    elif W.dim() == 3:  # (B, T, N) -> expand to match Y
        W = W.unsqueeze(-1)  # (B, T, N, 1)
    else:  # (B, 1, N)
        W = W.unsqueeze(-1).expand(B, T, N, 1)
    
    # Reshape X to (B, T, N, 3) by broadcasting
    X = X.expand(B, T, N, 3)
    
    # Compute weighted centroids
    sum_W = torch.sum(W, dim=2, keepdim=True)  # (B, T, 1, 1)
    centroid_X = torch.sum(W * X, dim=2) / sum_W.squeeze(-1)  # (B, T, 3)
    centroid_Y = torch.sum(W * Y, dim=2) / sum_W.squeeze(-1)  # (B, T, 3)
    
    # Center the point clouds
    X_centered = X - centroid_X.unsqueeze(2)  # (B, T, N, 3)
    Y_centered = Y - centroid_Y.unsqueeze(2)  # (B, T, N, 3)
    
    # Compute weighted covariance matrix H = X^T W Y
    X_weighted = X_centered * W  # (B, T, N, 3)
    H = torch.matmul(X_weighted.transpose(2, 3), Y_centered)  # (B, T, 3, 3)
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(H)  # U/Vt: (B, T, 3, 3)

    # Ensure right-handed rotation (det(R) = +1)
    det = torch.det(torch.matmul(U, Vt))  # (B, T)
    Vt_corrected = Vt.clone()
    mask = det < 0
    B_idx, T_idx = torch.nonzero(mask, as_tuple=True)
    Vt_corrected[B_idx, T_idx, -1, :] *= -1  # Flip last row for those needing correction

    # Optimal rotation and translation
    R = torch.matmul(U, Vt_corrected).inverse()  # (B, T, 3, 3)
    t = centroid_Y - torch.matmul(R, centroid_X.unsqueeze(-1)).squeeze(-1)  # (B, T, 3)
    w2c = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
    if (torch.det(R) - 1).abs().max() < 1e-3:
        w2c[:, :, :3, :3] = R
    else:
        import pdb; pdb.set_trace()
    w2c[:, :, :3, 3] = t
    try:
        c2w_traj = torch.inverse(w2c)  # or torch.linalg.inv()
    except:
        c2w_traj = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)

    return c2w_traj

def key_fr_wprocrustes(cam_pts, graph_matrix, dyn_weight, vis_mask,slide_len=16, overlap=8, K=3, mode="keyframe"):
    """
    cam_pts: (B, T, N, 3)
    graph_matrix: (B, 1, N)
    dyn_weight: (B, T, N)
    K: number of keyframes to select (including start and end)

    Returns:
        c2w_traj: (B, T, 4, 4)
    """
    B, T, N, _ = cam_pts.shape
    device = cam_pts.device

    if mode == "keyframe":
        # Step 1: Keyframe selection
        ky_fr_idx = [0, T - 1]
        graph_sum = torch.sum(graph_matrix, dim=-1)  # (B, T, T)
        dist = torch.max(graph_sum[:, 0, :], graph_sum[:, T - 1, :])  # (B, T)
        dist[:, [0, T - 1]] = float('inf')
        for _ in range(K - 2):  # already have 2
            last_idx = ky_fr_idx[-1]
            dist = torch.max(dist, graph_sum[:, last_idx, :])
            dist[:, last_idx] = float('inf')
            next_id = torch.argmin(dist, dim=1)[0].item()  # Assuming batch=1 or shared
            ky_fr_idx.append(next_id)

        ky_fr_idx = sorted(ky_fr_idx)
    elif mode == "slide":
        id_slide = torch.arange(0, T)
        id_slide = id_slide.unfold(0, slide_len, overlap)
        vis_mask_slide = vis_mask.unfold(1, slide_len, overlap)
        cam_pts_slide = cam_pts.unfold(1, slide_len, overlap)
        ky_fr_idx = torch.arange(0, T - slide_len + 1, overlap)
        if ky_fr_idx[-1] + slide_len < T:
            # if the last keyframe does not cover the whole sequence, add one more keyframe
            ky_fr_idx = torch.cat([ky_fr_idx, ky_fr_idx[-1:] + overlap])
            id_add = torch.arange(ky_fr_idx[-1], ky_fr_idx[-1] + slide_len).clamp(max=T-1)
            id_slide = torch.cat([id_slide, id_add[None, :]], dim=0)
            cam_pts_add = cam_pts[:, id_add, :, :]
            cam_pts_slide = torch.cat([cam_pts_slide, cam_pts_add.permute(0,2,3,1)[:, None, ...]], dim=1)
            vis_mask_add = vis_mask[:, id_add, :]
            vis_mask_slide = torch.cat([vis_mask_slide, vis_mask_add.permute(0,2,3,1)[:, None, ...]], dim=1)

    if mode == "keyframe":
        # Step 2: Weighted Procrustes in windows
        base_pose = torch.eye(4, device=cam_pts.device).view(1, 1, 4, 4).repeat(B, 1, 1, 1)  # (B, 1, 4, 4)
        c2w_traj_out = []
        for i in range(len(ky_fr_idx) - 1):
            start_idx = ky_fr_idx[i]
            end_idx = ky_fr_idx[i + 1]

            # Visibility mask
            vis_mask_i = graph_matrix[:, start_idx, end_idx, :]  # (B, N) or (N,)
            if vis_mask_i.dim() == 1:
                vis_mask_i = vis_mask_i.unsqueeze(0)  # (1, N)

            # Broadcast cam_pts and dyn_weight
            cam_ref = cam_pts[:, start_idx:start_idx+1, :, :]  # (B, 1, M, 3)
            cam_win = cam_pts[:, start_idx:end_idx+1, :, :]    # (B, W, M, 3)
            weight = dyn_weight[:, :, :] * vis_mask_i[:, None, :]     # (B, W, M)

            # Compute relative transformations
            if weight.sum() < 50:
                weight = weight.clamp(min=5e-2)
            relative_tfms = weighted_procrustes_torch(cam_ref, cam_win, weight)  # (B, W, 4, 4)

            # Apply to original c2w_traj
            updated_pose = base_pose.detach() @ relative_tfms                             # (B, W, 4, 4)
            base_pose = relative_tfms[:, -1:, :, :].detach()       # (B, 1, 4, 4)

            # Assign to output trajectory (avoid in-place on autograd path)
            c2w_traj_out.append(updated_pose[:, 1:, ...])
    
        c2w_traj_out = torch.cat(c2w_traj_out, dim=1)
        c2w_traj_out = torch.cat([torch.eye(4, device=device).repeat(B, 1, 1, 1), c2w_traj_out], dim=1)
    elif mode == "slide":
        c2w_traj_out = torch.eye(4, device=device).repeat(B, T, 1, 1)
        for i in range(cam_pts_slide.shape[1]):
            cam_pts_slide_i = cam_pts_slide[:, i, :, :].permute(0,3,1,2)
            id_slide_i = id_slide[i, :]
            vis_mask_i = vis_mask_slide[:, i, :, 0, :].permute(0,2,1)  # (B, N) or (N,)
            vis_mask_i = vis_mask_i[:,:1] * vis_mask_i
            weight_i = dyn_weight * vis_mask_i
            if weight_i.sum() < 50:
                weight_i = weight_i.clamp(min=5e-2)
            if i == 0:
                c2w_traj_out[:, id_slide_i, :, :] = weighted_procrustes_torch(cam_pts_slide_i[:,:1], cam_pts_slide_i, weight_i)
            else:
                campts_update = torch.einsum("btij,btnj->btni", c2w_traj_out[:,id_slide_i][...,:3,:3], cam_pts_slide_i) + c2w_traj_out[:,id_slide_i][...,None,:3,3]
                c2w_traj_update = weighted_procrustes_torch(campts_update[:,:1], campts_update, weight_i)
                c2w_traj_out[:, id_slide_i, :, :] = c2w_traj_update@c2w_traj_out[:,id_slide_i]

    return c2w_traj_out

def posenc(x, min_deg, max_deg):
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
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )

    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


class EfficientUpdateFormer3D(nn.Module):
    """
    Transformer model that updates track in 3D
    """

    def __init__(
        self,
        EFormer: EfficientUpdateFormer,
        update_points=True
    ):
        super().__init__()

        hidden_size =  EFormer.hidden_size       
        num_virtual_tracks = EFormer.num_virtual_tracks
        num_heads = EFormer.num_heads
        mlp_ratio = 4.0
 
        #NOTE: we design a switcher to bridege the camera pose, 3d tracks and 2d tracks 

        # iteract with pretrained 2d tracking
        self.switcher_tokens = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        # cross attention
        space_depth=len(EFormer.space_virtual_blocks)
        self.space_switcher_blocks = nn.ModuleList(
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
        
        # config 3d tracks blocks
        self.space_track3d2switcher_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(space_depth)
            ]
        )
        self.space_switcher2track3d_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(space_depth)
            ]
        )
        # config switcher blocks
        self.space_virtual2switcher_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(space_depth)
            ]
        )
        self.space_switcher2virtual_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(space_depth)
            ]
        )
        # config the temporal blocks
        self.time_blocks_new = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(len(EFormer.time_blocks))
            ]
        )
        # scale and shift cross attention
        self.scale_shift_cross_attn = nn.ModuleList(
            [
                CrossAttnBlock(
                    128, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(len(EFormer.time_blocks))
            ]
        )
        self.scale_shift_self_attn = nn.ModuleList(
            [
                AttnBlock(
                    128, num_heads, mlp_ratio=mlp_ratio, attn_class=Attention
                )
                for _ in range(len(EFormer.time_blocks))
            ]
        )       
        self.scale_shift_dec = torch.nn.Linear(128, 128+1, bias=True)

        # dense cross attention
        self.dense_res_cross_attn = nn.ModuleList(
            [
                CrossAttnBlock(
                    128, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(len(EFormer.time_blocks))
            ]
        )
        self.dense_res_self_attn = nn.ModuleList(
            [
                AttnBlock(
                    128, num_heads, mlp_ratio=mlp_ratio, attn_class=Attention
                )
                for _ in range(len(EFormer.time_blocks))
            ]
        )       
        self.dense_res_dec = torch.nn.Conv2d(128, 3+128, kernel_size=1, stride=1, padding=0)

        # set different heads
        self.update_points = update_points
        if update_points:
            self.point_head = torch.nn.Linear(hidden_size, 4, bias=True)
        else:
            self.depth_head = torch.nn.Linear(hidden_size, 1, bias=True)
        self.pro_analysis_w_head = torch.nn.Linear(hidden_size, 1, bias=True)
        self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        self.residual_head = torch.nn.Linear(hidden_size,
                                              hidden_size, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if getattr(self, "point_head", None) is not None:
                torch.nn.init.trunc_normal_(self.point_head.weight, std=1e-6)
                torch.nn.init.constant_(self.point_head.bias, 0)
            if getattr(self, "depth_head", None) is not None:   
                torch.nn.init.trunc_normal_(self.depth_head.weight, std=0.001)
            if getattr(self, "vis_conf_head", None) is not None:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=1e-6)
            if getattr(self, "scale_shift_dec", None) is not None:
                torch.nn.init.trunc_normal_(self.scale_shift_dec.weight, std=0.001)
            if getattr(self, "residual_head", None) is not None:
                torch.nn.init.trunc_normal_(self.residual_head.weight, std=0.001)
            

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        
    def forward(self, input_tensor, input_tensor3d, EFormer: EfficientUpdateFormer,
                        mask=None, add_space_attn=True, extra_sparse_tokens=None, extra_dense_tokens=None):
        
        #NOTE: prepare the pose and 3d tracks features
        tokens3d = EFormer.input_transform(input_tensor3d)
        
        tokens = EFormer.input_transform(input_tensor)
        B, _, T, _ = tokens.shape
        virtual_tokens = EFormer.virual_tracks.repeat(B, 1, T, 1)
        switcher_tokens = self.switcher_tokens.repeat(B, 1, T, 1)
        
        tokens = torch.cat([tokens, virtual_tokens], dim=1)
        tokens3d = torch.cat([tokens3d, switcher_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        layers = []
            

        for i in range(len(EFormer.time_blocks)):
            if extra_sparse_tokens is not None:
                extra_sparse_tokens = rearrange(extra_sparse_tokens, 'b n t c -> (b t) n c')
                extra_sparse_tokens = self.scale_shift_cross_attn[i](extra_sparse_tokens, rearrange(tokens3d, 'b n t c -> (b t) n c'))
                extra_sparse_tokens = rearrange(extra_sparse_tokens, '(b t) n c -> (b n) t c', b=B, t=T)
                extra_sparse_tokens = self.scale_shift_self_attn[i](extra_sparse_tokens)
                extra_sparse_tokens = rearrange(extra_sparse_tokens, '(b n) t c -> b n t c', b=B, n=2, t=T)
            
            if extra_dense_tokens is not None:
                h_p, w_p = extra_dense_tokens.shape[-2:]
                extra_dense_tokens = rearrange(extra_dense_tokens, 'b t c h w -> (b t) (h w) c')
                extra_dense_tokens = self.dense_res_cross_attn[i](extra_dense_tokens, rearrange(tokens3d, 'b n t c -> (b t) n c'))
                extra_dense_tokens = rearrange(extra_dense_tokens, '(b t) n c -> (b n) t c', b=B, t=T)
                extra_dense_tokens = self.dense_res_self_attn[i](extra_dense_tokens)
                extra_dense_tokens = rearrange(extra_dense_tokens, '(b h w) t c -> b t c h w', b=B, h=h_p, w=w_p)
            
            # temporal
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = EFormer.time_blocks[i](time_tokens)

            # temporal 3d
            time_tokens3d = tokens3d.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens3d = self.time_blocks_new[i](time_tokens3d)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            tokens3d = time_tokens3d.view(B, N, T, -1)

            if (
                add_space_attn
                and hasattr(EFormer, "space_virtual_blocks")
                and (i % (len(EFormer.time_blocks) // len(EFormer.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C
                space_tokens3d = (
                    tokens3d.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C

                point_tokens = space_tokens[:, : N - EFormer.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - EFormer.num_virtual_tracks :]
                # get the 3d relevant tokens
                track3d_tokens = space_tokens3d[:, : N - EFormer.num_virtual_tracks]
                switcher_tokens = space_tokens[:, N - EFormer.num_virtual_tracks + 1:]

                # iteract switcher with pose and tracks3d
                switcher_tokens = self.space_track3d2switcher_blocks[j](
                    switcher_tokens, track3d_tokens, mask=mask
                )

                
                virtual_tokens = EFormer.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )

                # get the switcher_tokens
                switcher_tokens = self.space_virtual2switcher_blocks[j](
                    switcher_tokens, virtual_tokens
                )
                virtual_tokens_res = self.residual_head(
                    self.space_switcher2virtual_blocks[j](
                    virtual_tokens, switcher_tokens
                )
                )
                switcher_tokens_res = self.residual_head(
                    self.space_switcher2virtual_blocks[j](
                    switcher_tokens, virtual_tokens
                )
                )
                # add residual 
                virtual_tokens = virtual_tokens + virtual_tokens_res
                switcher_tokens = switcher_tokens + switcher_tokens_res

                virtual_tokens = EFormer.space_virtual_blocks[j](virtual_tokens)
                switcher_tokens = self.space_switcher_blocks[j](switcher_tokens)
                # decode
                point_tokens = EFormer.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )
                track3d_tokens = self.space_switcher2track3d_blocks[j](
                    track3d_tokens, switcher_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                space_tokens3d =  torch.cat([track3d_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                tokens3d = space_tokens3d.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C

                j += 1

        tokens = tokens[:, : N - EFormer.num_virtual_tracks]
        track3d_tokens = tokens3d[:, : N - EFormer.num_virtual_tracks]

        if self.update_points:
            depth_update, dynamic_prob_update = self.point_head(track3d_tokens)[..., :3], self.point_head(track3d_tokens)[..., 3:] 
        else:
            depth_update, dynamic_prob_update = self.depth_head(track3d_tokens)[..., :1], self.depth_head(track3d_tokens)[..., 1:] 
        pro_analysis_w = self.pro_analysis_w_head(track3d_tokens)
        flow = EFormer.flow_head(tokens)
        if EFormer.linear_layer_for_vis_conf:
            vis_conf = EFormer.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)
        if extra_sparse_tokens is not None:
            scale_shift_out = self.scale_shift_dec(extra_sparse_tokens)
            dense_res_out = self.dense_res_dec(extra_dense_tokens.view(B*T, -1, h_p, w_p)).view(B, T, -1, h_p, w_p)
            return flow, depth_update, dynamic_prob_update, pro_analysis_w, scale_shift_out, dense_res_out
        else:
            return flow, depth_update, dynamic_prob_update, pro_analysis_w, None, None

def recover_global_translations_batch(global_rot, c2w_traj, graph_weight):
    B, T = global_rot.shape[:2]
    device = global_rot.device

    # Compute R_i @ t_ij
    t_rel = c2w_traj[:, :, :, :3, 3]  # (B, T, T, 3)
    R_i = global_rot[:, :, None, :, :]  # (B, T, 1, 3, 3)
    t_rhs = torch.matmul(R_i, t_rel.unsqueeze(-1)).squeeze(-1)  # (B, T, T, 3)

    # Mask: exclude self-loops and small weights
    valid_mask = (graph_weight > 1e-5) & (~torch.eye(T, dtype=bool, device=device)[None, :, :])  # (B, T, T)

    # Get all valid (i, j) edge indices
    i_idx, j_idx = torch.meshgrid(
        torch.arange(T, device=device),
        torch.arange(T, device=device),
        indexing="ij"
    )
    i_idx = i_idx.reshape(-1)  # (T*T,)
    j_idx = j_idx.reshape(-1)

    # Expand to batch (B, T*T)
    i_idx = i_idx[None, :].repeat(B, 1)
    j_idx = j_idx[None, :].repeat(B, 1)

    # Flatten everything
    valid_mask_flat = valid_mask.view(B, -1)  # (B, T*T)
    w_flat = graph_weight.view(B, -1)         # (B, T*T)
    rhs_flat = t_rhs.view(B, -1, 3)           # (B, T*T, 3)

    # Initialize output translations
    global_translations = torch.zeros(B, T, 3, device=device)

    for b_id in range(B):
        mask = valid_mask_flat[b_id]
        i_valid = i_idx[b_id][mask]
        j_valid = j_idx[b_id][mask]
        w_valid = w_flat[b_id][mask]
        rhs_valid = rhs_flat[b_id][mask]

        n_edges = i_valid.shape[0]
        
        # Build A matrix: (n_edges*3, T*3)
        A = torch.zeros(n_edges*3, T*3, device=device)
        
        # Build b vector: (n_edges*3,)
        b = torch.zeros(n_edges*3, device=device)
        
        for k in range(n_edges):
            i, j = i_valid[k], j_valid[k]
            weight = w_valid[k]
            
            # Fill A matrix for x,y,z components
            for dim in range(3):
                row = k*3 + dim
                A[row, i*3 + dim] = -weight
                A[row, j*3 + dim] = weight
                
                # Fill b vector
                b[row] = rhs_valid[k, dim] * weight

        # Solve least squares
        try:
            # Add small regularization for stability
            AtA = A.transpose(-1, -2) @ A + 1e-4 * torch.eye(A.shape[-1], device=A.device)
            Atb = A.transpose(-1, -2) @ b.unsqueeze(-1)
            
            solution = torch.linalg.solve(AtA, Atb).squeeze(-1)  # (3*T,)
            t_batch = solution.view(T, 3)

            # Fix scale by setting first frame to origin
            t_batch = t_batch - t_batch[0:1]
            global_translations[b_id] = t_batch

        except RuntimeError as e:
            print(f"Error in batch {b_id}: {e}")
            global_translations[b_id] = torch.zeros(T, 3, device=device)
    return global_translations


def global_graph_motion_average(c2w_traj, graph_weight):
    """
    This function will average the c2w_traj by the graph_weight
    """
    B, T, T, _, _ = c2w_traj.shape
    mask = graph_weight[..., 0, 0] < 1e-5  # (B, T, T)
    mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 4, 4)  # (B, T, T, 4, 4)
    identity = torch.eye(4, device=c2w_traj.device).view(1, 1, 1, 4, 4).expand(B, T, T, 4, 4)
    c2w_traj = torch.where(mask, identity, c2w_traj)

    Rot_rel_weighted = c2w_traj[:,:,:,:3,:3].contiguous() * graph_weight     # B T T 3 3
    Rot_big = Rot_rel_weighted.permute(0, 1, 3, 2, 4).reshape(B, 3*T, 3*T) # B 3T 3T
    epsilon = 1e-8
    I_big = torch.eye(3*T, device=Rot_big.device).unsqueeze(0)  # (1, 3T, 3T)
    Rot_big_reg = Rot_big + epsilon * I_big  # (B, 3T, 3T)
    #NOTE: cal the global rotation
    # Step 1: batch eigendecomposition
    try:
        eigvals, eigvecs = torch.linalg.eigh(Rot_big_reg)  # eigvecs: (B, 3T, 3)
    except:
        import pdb; pdb.set_trace()
    # Step 2: get the largest 3 eigenvectors
    X = eigvecs[:, :, -3:]  # (B, 3T, 3)
    # Step 3: split into (B, T, 3, 3)
    X = X.view(B, T, 3, 3)  # each frame's rotation block (non-orthogonal)
    # Step 4: project to SO(3), using SVD
    U, _, Vh = torch.linalg.svd(X)  # (B, T, 3, 3)
    R = U @ Vh
    # Step 5: ensure det(R)=1 (right-handed coordinate system)
    det = torch.linalg.det(R)  # (B, T)
    neg_det_mask = det < 0
    # if det<0, reverse the last column and multiply
    U_flip = U.clone()
    U_flip[neg_det_mask, :, -1] *= -1
    R = U_flip @ Vh
    # global rotation    
    Rot_glob = R[:,:1].inverse() @ R
    # global translation
    t_glob = recover_global_translations_batch(Rot_glob,
                                                c2w_traj, graph_weight[...,0,0])
    c2w_traj_final = torch.eye(4, device=c2w_traj.device)[None,None].repeat(B, T, 1, 1)
    c2w_traj_final[:,:,:3,:3] = Rot_glob
    c2w_traj_final[:,:,:3,3] = t_glob
    
    return c2w_traj_final


def depth_to_points_colmap(metric_depth: torch.Tensor,
                           intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Unproject a depth map to a point cloud in COLMAP convention.

    Args:
        metric_depth: (B, H, W) depth map, meters.
        intrinsics:   (B, 3, 3) COLMAP-style K matrix.
    Returns:
        points_map:   (B, H, W, 3) point cloud in camera coordinates.
    """
    # 因为输入的 metric_depth 维度是 (B, H, W)
    B, H, W = metric_depth.shape

    # 因为需要每个像素的 [u, v, 1] 齐次坐标
    u = torch.arange(W, device=metric_depth.device, dtype=metric_depth.dtype)
    v = torch.arange(H, device=metric_depth.device, dtype=metric_depth.dtype)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
    pix = pix.reshape(-1, 3)  # (H*W, 3)
    # 因为要对 B 张图做相同操作
    pix = pix.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 3)
    # import pdb; pdb.set_trace()
    # 因为 K 是 (B, 3, 3)
    K_inv = torch.inverse(intrinsics)        # (B, 3, 3)

    # 因为反投影方向是 X_cam = K^{-1} * pix
    dirs = torch.einsum('bij,bkj->bki', K_inv, pix)  # (B, H*W, 3)

    # 因为要按深度伸缩
    depths = metric_depth.reshape(B, -1)           # (B, H*W)
    pts = dirs * depths.unsqueeze(-1)              # (B, H*W, 3)

    # 因为希望输出 (B, H, W, 3)
    points_map = pts.view(B, H, W, 3)              # (B, H, W, 3)

    return points_map

def vec6d_to_R(vector_6D):
    v1=vector_6D[:,:3]/vector_6D[:,:3].norm(dim=-1,keepdim=True)
    v2=vector_6D[:,3:]-(vector_6D[:,3:]*v1).sum(dim=-1,keepdim=True)*v1
    v2=v2/v2.norm(dim=-1,keepdim=True)
    v3=torch.cross(v1,v2,dim=-1)
    return torch.concatenate((v1.unsqueeze(1),v2.unsqueeze(1),v3.unsqueeze(1)),dim=1)

class MyTransformerHead(nn.Module):
    def __init__(self,input_dim,dim,use_positional_encoding_transformer):
        super(MyTransformerHead,self).__init__()
       
        patch_dim=input_dim+1
        self.layers=3
        # dim=128
        self.use_positional_encoding_transformer=use_positional_encoding_transformer
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.transformer_frames=[]
        self.transformer_points=[]
        
        for i in range(self.layers):
            self.transformer_frames.append(Transformer(dim, 1, 16, 64, 2048))
            self.transformer_points.append(Transformer(dim, 1, 16, 64, 2048))
        self.transformer_frames=nn.ModuleList(self.transformer_frames)
        self.transformer_points=nn.ModuleList(self.transformer_points)

    def forward(self, x):
        
        
        x=torch.cat((x,torch.ones(x.shape[0],x.shape[1],1,x.shape[3]).cuda()),dim=2)
        
        x=x.transpose(2,3)
        
        b,n,f,c=x.shape
        x=self.to_patch_embedding(x)
        
        x=x.view(b*n,f,-1) # x.shape [390, 33, 256]
        if self.use_positional_encoding_transformer:
            pe = posemb_sincos_1d(x) #pe.shape= [33,256] (33 frame, 256 embedding dim)
            x=pe.unsqueeze(0)+x 
        for i in range(self.layers):
            #frames aggregation
            x=self.transformer_frames[i](x)
            
            #point sets aggregation
            x=x.view(b,n,f,-1).transpose(1,2).reshape(b*f,n,-1) 
            
            x=self.transformer_points[i](x)

            x=x.view(b,f,n,-1)
            x=x.transpose(1,2).reshape(b*n,f,-1)

        x=x.view(b,n,f,-1)
        x=x.transpose(2,3)

       
        return x

def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output

class TrackFusion(nn.Module):
    def __init__(self,width1=320,conv2_kernel_size=31,K=12,
                                    conv_kernel_size=3,inputdim=2,use_positionl_encoding=True,
                                    positional_dim=4,use_transformer=True,detach_cameras_dynamic=True,
                                    use_positional_encoding_transformer=True,use_set_of_sets=False,predict_focal_length=False):
        super(TrackFusion, self).__init__()
        self.predict_focal_length=predict_focal_length
        self.inputdim = inputdim
        self.n1 = width1

        self.K=K
        self.n2 = 6+3+1+self.K+2
        self.detach_cameras_dynamic=detach_cameras_dynamic
        l=conv_kernel_size
        # layers
        self.use_set_of_sets=use_set_of_sets
        self.use_positionl_encoding=use_positionl_encoding
        self.positional_dim=positional_dim
        actual_input_dim=inputdim
        if self.use_positionl_encoding:
            actual_input_dim=2 * inputdim * self.positional_dim+inputdim

        self.use_transformer=use_transformer
        
        if self.use_positionl_encoding:
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(self.positional_dim)],requires_grad = False)
        
        if True:
            if self.use_transformer:
                self.transformer_my=MyTransformerHead(actual_input_dim,width1,use_positional_encoding_transformer)
          
            self.conv_final = nn.Conv1d(self.n1, self.n2, kernel_size=conv2_kernel_size,stride=1, padding=conv2_kernel_size//2, padding_mode='circular')
            
            self.fc1 = nn.Linear(self.n1,3*self.K+1)



            torch.nn.init.xavier_uniform_(self.conv_final.weight)

            torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x, pts_miu=None, pts_radis=None, simple_return=True):

        B, N, C, T = x.shape
        if self.use_positionl_encoding:
            x_original_shape=x.shape
            x=x.transpose(2,3)
            x=x.reshape(-1,x.shape[-1])
            if self.b.device!=x.device:
                self.b=self.b.to(x.device)
            pos = positionalEncoding_vec(x,self.b)
            x=torch.cat((x,pos),dim=1)
            x=x.view(x_original_shape[0],x_original_shape[1],x_original_shape[3],x.shape[-1]).transpose(2,3)

        b = len(x)
        n= x.shape[1]
        l= x.shape[-1]
        if self.use_set_of_sets:
            cameras,perpoint_features=self.set_of_sets_my(x)
        else:
            if  self.use_transformer:
                x=self.transformer_my(x)
            else:
                for i in range(len( self.conv1)):
                    if i==0:
                        x = x.reshape(n*b, x.shape[2],l)
                    else:
                        x = x.view(n * b, self.n1, l)
                    x1 = self.bn1[i](self.conv1[i](x)).view(b,n,self.n1,l)
                    x2 = self.bn1s[i](self.conv1s[i](x)).view(b,n,self.n1,l).mean(dim=1).view(b,1,self.n1,l).repeat(1,n,1,1)
                    x = F.relu(x1 + x2)

            cameras=torch.mean(x,dim=1) 
            cameras=self.conv_final(cameras)
            perpoint_features = torch.mean(x,dim=3)
            perpoint_features = self.fc1(perpoint_features.view(n*b,self.n1))

        B=perpoint_features[:,:self.K*3].view(b,n,3,self.K)     # motion basis
        NR=F.elu(perpoint_features[:,-1].view(b,n))+1+0.00001   

        position_params=cameras[:,:3,:]
        if self.predict_focal_length:
            focal_params=1+0.05*cameras[:,3:4,:].clone().transpose(1,2)
        else:
            focal_params=1.0
        basis_params=cameras[:,4:4+self.K]
        basis_params[:,0,:]=torch.clamp(basis_params[:,0,:].clone(),min=1.0,max=1.0)
        basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1)
        rotation_params=cameras[:,4+self.K:4+self.K+6]
        # Converting rotation parameters into a valid rotation matrix (probably better to move to 6d representation)
        rotation_params=vec6d_to_R(rotation_params.transpose(1,2).reshape(b*l,6)).view(b,l,3,3)
        
        # Transfering global 3D into each camera coordinates (using per camera roation and translation)
        points3D_static=((basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1))[:,:,:,:,:1]*B.unsqueeze(-2)[:,:,:,:,:1]).sum(-1)
        
        if  self.detach_cameras_dynamic==False:
            points3D=((basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1))[:,:,:,:,1:]*B.unsqueeze(-2)[:,:,:,:,1:]).sum(-1)+points3D_static
        else:
            points3D=((basis_params.transpose(1,2).unsqueeze(1).unsqueeze(1))[:,:,:,:,1:]*B.unsqueeze(-2)[:,:,:,:,1:]).sum(-1)+points3D_static.detach()

        points3D=points3D.transpose(1,3)
        points3D_static=points3D_static.transpose(1,3)
        position_params=position_params.transpose(1,2)
        if pts_miu is not None:
            position_params=position_params*pts_radis.squeeze(-1)+pts_miu.squeeze(-2)
            points3D_static = points3D_static*pts_radis.squeeze(-1)+pts_miu.permute(0,1,3,2)
            points3D = points3D*pts_radis.squeeze(-1)+pts_miu.permute(0,1,3,2)
        
        if  self.detach_cameras_dynamic==False:
            points3D_camera=(torch.bmm(rotation_params.view(b*l,3,3).transpose(1,2),points3D.reshape(b*l,3,n)-position_params.reshape(b*l,3).unsqueeze(-1)))
            points3D_camera=points3D_camera.view(b,l,3,n)
        else:
            points3D_camera=(torch.bmm(rotation_params.view(b*l,3,3).transpose(1,2).detach(),points3D.reshape(b*l,3,n)-position_params.detach().reshape(b*l,3).unsqueeze(-1)))
            points3D_camera=points3D_camera.view(b,l,3,n)
        points3D_static_camera=(torch.bmm(rotation_params.view(b*l,3,3).transpose(1,2),points3D_static.reshape(b*l,3,n)-position_params.reshape(b*l,3).unsqueeze(-1)))
        points3D_static_camera=points3D_static_camera.view(b,l,3,n)
      
        # Projecting from 3D to 2D
        projections=points3D_camera.clone()
        projections_static=points3D_static_camera.clone()
        
        depths=projections[:,:,2,:]
        depths_static=projections_static[:,:,2,:]

        projectionx=focal_params*projections[:,:,0,:]/torch.clamp(projections[:,:,2,:].clone(),min=0.01)
        projectiony=focal_params*projections[:,:,1,:]/torch.clamp(projections[:,:,2,:].clone(),min=0.01)

        projectionx_static=focal_params*projections_static[:,:,0,:]/torch.clamp(projections_static[:,:,2,:].clone(),min=0.01)
        projectiony_static=focal_params*projections_static[:,:,1,:]/torch.clamp(projections_static[:,:,2,:].clone(),min=0.01)
         
        projections2=torch.cat((projectionx.unsqueeze(2),projectiony.unsqueeze(2)),dim=2)
        projections2_static=torch.cat((projectionx_static.unsqueeze(2),projectiony_static.unsqueeze(2)),dim=2)
                
        if simple_return:
            c2w_traj = torch.eye(4, device=x.device)[None,None].repeat(b,T,1,1)
            c2w_traj[:,:,:3,:3] = rotation_params
            c2w_traj[:,:,:3,3] = position_params
            return c2w_traj, points3D, points3D_camera
        else:
            return focal_params,projections2,projections2_static,rotation_params,position_params,B,points3D,points3D_static,depths,depths_static,0,basis_params,0,0,points3D_camera,NR


def get_nth_visible_time_index(vis_gt: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """
    vis_gt: [B, T, N]  0/1 binary tensor
    n: [B, N] int tensor, the n-th visible time index to get (1-based)
    Returns: [B, N] tensor of time indices into T, or -1 if not enough visible steps
    """
    B, T, N = vis_gt.shape

    # Create a tensor [0, 1, ..., T-1] for time indices
    time_idx = torch.arange(T, device=vis_gt.device).view(1, T, 1).expand(B, T, N)  # [B, T, N]

    # Mask invisible steps with a large number (T)
    masked_time = torch.where(vis_gt.bool(), time_idx, torch.full_like(time_idx, T))

    # Sort along time dimension
    sorted_time, _ = masked_time.sort(dim=1)  # [B, T, N]

    # Prepare index tensor for gather: [B, N] -> [B, 1, N]
    gather_idx = (n - 1).clamp(min=0, max=T-1).unsqueeze(1)  # shape: [B, 1, N]
    assert gather_idx.shape == sorted_time.shape[:1] + (1, sorted_time.shape[2])  # [B, 1, N]
    
    # Gather from sorted_time: result is [B, 1, N]
    nth_time = sorted_time.gather(dim=1, index=gather_idx).squeeze(1)  # [B, N]

    # If value is T (i.e., masked), then not enough visible → set to -1
    nth_time = torch.where(nth_time == T, torch.full_like(nth_time, -1), nth_time)

    return nth_time  # [B, N]

def knn_torch(x, k):
    """
    x: (B, T, N, 2)
    return: indices of k-NN, shape (B, T, N, k)
    """
    B, T, N, C = x.shape
    # Reshape to (B*T, N, 2)
    x = x.view(B*T, N, C)  # Merge the first two dimensions for easier processing
    # Calculate pairwise distance: (B*T, N, N)
    dist = torch.cdist(x, x, p=2)  # Euclidean distance

    # Exclude self: set diagonal to a large number (to prevent self from being a neighbor)
    mask = torch.eye(N, device=x.device).bool()[None, :, :]  # (1, N, N)
    dist.masked_fill_(mask, float('inf'))

    # Get indices of top k smallest distances
    knn_idx = dist.topk(k, largest=False).indices  # (B*T, N, k)
    # Restore dimensions (B, T, N, k)
    knn_idx = knn_idx.view(B, T, N, k)
    return knn_idx

def get_topo_mask(coords_xyz_append: torch.Tensor,
                                 coords_2d_lift: torch.Tensor, replace_ratio: float = 0.6) -> torch.Tensor:
    """
    coords_xyz_append: [B, T, N, 3] 3d coordinates
    coords_2d_lift: [B*T, N] depth map
    replace_ratio: float, the ratio of the depth change to be considered as a topological change
    """
    B, T, N, _ = coords_xyz_append.shape
    # if N > 1024:
    #     pick_idx = torch.randperm(N)[:1024]
    # else:
    pick_idx = torch.arange(N, device=coords_xyz_append.device)
    coords_xyz_append = coords_xyz_append[:,:,pick_idx,:]
    knn_idx = knn_torch(coords_xyz_append, 49)
    knn_idx = pick_idx[knn_idx]
    # raw topology
    raw_depth = coords_xyz_append[...,2:]   # B T N 1     knn_idx  B T N K
    knn_depth = torch.gather(
            raw_depth.expand(-1, -1, -1, knn_idx.shape[-1]),  # (B, T, N, K)
            dim=2,
            index=knn_idx  # (B, T, N, K)
        ).squeeze(-1)  # → (B, T, N, K)
    depth_rel_neg_raw = (knn_depth - raw_depth)
    # unproj depth
    knn_depth_unproj = torch.gather(
            depth_unproj.view(B,T,N,1).expand(-1, -1, -1, knn_idx.shape[-1]),  # (B, T, N, K)
            dim=2,
            index=knn_idx  # (B, T, N, K)
        ).squeeze(-1)  # → (B, T, N, K)
    depth_rel_neg_unproj = (knn_depth_unproj - depth_unproj.view(B,T,N,1))
    # topological change threshold
    mask_topo = (depth_rel_neg_raw.abs() / (depth_rel_neg_unproj.abs()+1e-8) - 1).abs() < 0.4
    mask_topo = mask_topo.sum(dim=-1) > 9
    
    return mask_topo
    
    
