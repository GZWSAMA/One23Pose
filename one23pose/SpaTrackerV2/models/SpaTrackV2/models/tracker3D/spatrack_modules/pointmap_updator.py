import torch
import torch.nn as nn
from models.SpaTrackV2.models.blocks import bilinear_sampler
from models.SpaTrackV2.models.tracker3D.spatrack_modules.alignment import align_points_scale, align_points_scale_xyz_shift

def compute_affine_scale_and_shift(points, pointmap, mask, weights=None, eps=1e-6):
    """
    Compute global affine transform (scale * pointmap + shift = points)
    using least-squares fitting with optional weights and mask.

    Args:
        points (BT, N, 3): Target points
        pointmap (BT, N, 3): Source points
        mask (BT, N): Binary mask indicating valid points
        weights (BT, N): Optional weights per point
        eps (float): Numerical stability

    Returns:
        scale (BT, 1): Scalar scale per batch
        shift (BT, 3): Shift vector per batch
    """
    if weights is None:
        weights = mask.float()
    else:
        weights = weights * mask  # combine mask

    # Sum of weights
    weight_sum = weights.sum(dim=1, keepdim=True) + eps  # (BT, 1)

    # Compute weighted centroids
    centroid_p = (points * weights.unsqueeze(-1)).sum(dim=1) / weight_sum  # (BT, 3)
    centroid_m = (pointmap * weights.unsqueeze(-1)).sum(dim=1) / weight_sum  # (BT, 3)

    # Center the point sets
    p_centered = points - centroid_p.unsqueeze(1)  # (BT, N, 3)
    m_centered = pointmap - centroid_m.unsqueeze(1)  # (BT, N, 3)

    # Compute scale: ratio of dot products
    numerator = (weights.unsqueeze(-1) * (p_centered * m_centered)).sum(dim=1).sum(dim=-1)  # (BT,)
    denominator = (weights.unsqueeze(-1) * (m_centered ** 2)).sum(dim=1).sum(dim=-1) + eps  # (BT,)
    scale = (numerator / denominator).unsqueeze(-1)  # (BT, 1)

    # Compute shift: t = c_p - s * c_m
    shift = centroid_p - scale * centroid_m  # (BT, 3)

    return scale, shift

def compute_weighted_std(track2d, vis_est, eps=1e-6):
    """
    Compute the weighted standard deviation of 2D tracks across time.

    Args:
        track2d (Tensor): shape (B, T, N, 2), 2D tracked points.
        vis_est (Tensor): shape (B, T, N), visibility weights (0~1).
        eps (float): small epsilon to avoid division by zero.

    Returns:
        std (Tensor): shape (B, N, 2), weighted standard deviation for each point.
    """
    B, T, N, _ = track2d.shape

    # Compute weighted mean
    weighted_sum = (track2d * vis_est[..., None]).sum(dim=1)  # (B, N, 2)
    weight_sum = vis_est.sum(dim=1)[..., None] + eps          # (B, N, 1)
    track_mean = weighted_sum / weight_sum                    # (B, N, 2)

    # Compute squared residuals
    residuals = track2d - track_mean[:, None, :, :]           # (B, T, N, 2)
    weighted_sq_res = (residuals ** 2) * vis_est[..., None]   # (B, T, N, 2)

    # Compute weighted variance and std
    var = weighted_sq_res.sum(dim=1) / (weight_sum + eps)     # (B, N, 2)
    std = var.sqrt()                                           # (B, N, 2)

    return std

class PointMapUpdator(nn.Module):
    def __init__(self, stablizer):
        super(PointMapUpdator, self).__init__()
        self.stablizer = stablizer()

    def init_pointmap(self, points_map):

        pass
    
    def scale_update_from_tracks(self, cam_pts_est, coords_append, point_map_org, vis_est, reproj_loss):
        B, T, N, _ = coords_append.shape
        track2d = coords_append[...,:2].view(B*T, N, 2)
        
        track_len_std = compute_weighted_std(track2d.view(B, T, N, 2), vis_est.view(B, T, N)).norm(dim=-1)
        
        point_samp = bilinear_sampler(point_map_org, track2d[:,None], mode="nearest")
        point_samp = point_samp.permute(0,3,1,2).view(B*T, N, 3)
        cam_pts_est = cam_pts_est.view(B*T, N, 3)
        # mask 
        mask = vis_est.view(B*T, N)
        # using gaussian weights, mean is 2 pixels
        nm_reproj_loss = (reproj_loss.view(B*T, N) / (track_len_std.view(B, N) + 1e-6)).clamp(0, 5)
        std = nm_reproj_loss.std(dim=-1).view(B*T, 1) # B*T 1
        weights = torch.exp(-(0.5-nm_reproj_loss.view(B*T, N))**2 / (2*std**2))
        mask = mask*(point_samp[...,2]>0)*(cam_pts_est[...,2]>0)*weights
        scales, shift = align_points_scale_xyz_shift(point_samp, cam_pts_est, mask)

        return scales, shift