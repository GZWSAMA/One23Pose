# This file contains the loss functions for FrontTracker

import torch
import torch.nn as nn
import utils3d
from models.moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss, 
    edge_loss,
    normal_loss, 
    mask_l2_loss, 
    mask_bce_loss,
    monitoring, 
)
import torch.nn.functional as F
from models.SpaTrackV2.models.utils import pose_enc2mat, matrix_to_quaternion, get_track_points, normalize_rgb
from models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import depth_to_points_colmap, get_nth_visible_time_index
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding

def compute_loss(predictions, annots):
    """
    Compute the loss for the FrontTracker model.
    """

    B, T, C, H, W = predictions["images"].shape
    H_resize, W_resize = H, W

    if "poses_gt" in annots.keys():
        intrs, c2w_traj_gt = pose_enc2mat(annots["poses_gt"],
                                    H_resize, W_resize, min(H, W))
    else:
        c2w_traj_gt = None

    if "intrs_gt" in annots.keys():
        intrs = annots["intrs_gt"].view(B, T, 3, 3)
        fx_factor = W_resize / W
        fy_factor = H_resize / H
        intrs[:,:,0,:] *= fx_factor
        intrs[:,:,1,:] *= fy_factor
        
    if "depth_gt" in annots.keys():
        
        metric_depth_gt = annots['depth_gt'].view(B*T, 1, H, W)
        metric_depth_gt = F.interpolate(metric_depth_gt, 
                        size=(H_resize, W_resize), mode='nearest')
        
        _depths = metric_depth_gt[metric_depth_gt > 0].reshape(-1)
        q25 = torch.kthvalue(_depths, int(0.25 * len(_depths))).values
        q75 = torch.kthvalue(_depths, int(0.75 * len(_depths))).values
        iqr = q75 - q25
        upper_bound = (q75 + 0.8*iqr).clamp(min=1e-6, max=10*q25)
        _depth_roi = torch.tensor(
            [1e-1, upper_bound.item()], 
            dtype=metric_depth_gt.dtype, 
            device=metric_depth_gt.device
        )
        mask_roi = (metric_depth_gt > _depth_roi[0]) & (metric_depth_gt < _depth_roi[1])
        # fin mask
        gt_mask_fin = ((metric_depth_gt > 0)*(mask_roi)).float()
        # filter the sky
        inf_thres = 50*q25.clamp(min=200, max=1e3)
        gt_mask_inf = (metric_depth_gt > inf_thres).float()
        # gt mask
        gt_mask = (metric_depth_gt > 0)*(metric_depth_gt < 10*q25)
    
    points_map_gt = depth_to_points_colmap(metric_depth_gt.squeeze(1), intrs.view(B*T, 3, 3))
    
    if annots["syn_real"] == 1:
        ln_msk_l2, _ = mask_l2_loss(predictions["unc_metric"], gt_mask_fin[:,0], gt_mask_inf[:,0])
        ln_msk_l2 = 50*ln_msk_l2.mean()
    else:
        ln_msk_l2 = 0 * points_map_gt.mean()
    
    # loss1: global invariant loss
    ln_depth_glob, _, gt_metric_scale, gt_metric_shift = affine_invariant_global_loss(predictions["points_map"], points_map_gt, gt_mask[:,0], align_resolution=32)
    ln_depth_glob = 100*ln_depth_glob.mean()
    # loss2: edge loss
    ln_edge, _ = edge_loss(predictions["points_map"], points_map_gt, gt_mask[:,0])
    ln_edge = ln_edge.mean()
    # loss3: normal loss
    ln_normal, _ = normal_loss(predictions["points_map"], points_map_gt, gt_mask[:,0])
    ln_normal = ln_normal.mean()
    #NOTE: loss4: consistent loss
    norm_rescale = gt_metric_scale.mean()
    points_map_gt_cons = points_map_gt.clone() / norm_rescale
    if "scale" in predictions.keys():
        scale_ = predictions["scale"].view(B*T, 2, 1, 1)[:,:1]
        shift_ = predictions["scale"].view(B*T, 2, 1, 1)[:,1:]
    else:
        scale_ = torch.ones_like(predictions["points_map"])
        shift_ = torch.zeros_like(predictions["points_map"])[..., 2:]
    
    points_pred_cons = predictions["points_map"] * scale_ 
    points_pred_cons[..., 2:] += shift_
    pred_mask = predictions["unc_metric"].clone().clamp(min=5e-2)
    ln_cons = torch.abs(points_pred_cons - points_map_gt_cons).norm(dim=-1) * pred_mask - 0.05 * torch.log(pred_mask)
    ln_cons = 0.5*ln_cons[(1-gt_mask_inf.squeeze()).bool()].clamp(max=100).mean()
    # loss5: scale shift loss
    if "scale" in predictions.keys():
        ln_scale_shift = torch.abs(scale_.squeeze() - gt_metric_scale / norm_rescale) + torch.abs(shift_.squeeze() - gt_metric_shift[:,2] / norm_rescale)
        ln_scale_shift = 10*ln_scale_shift.mean()
    else:
        ln_scale_shift = 0 * ln_cons.mean()
    # loss6: pose loss
    c2w_traj_gt[...,:3, 3] /= norm_rescale
    ln_pose = 0
    for i_t, pose_enc_i in enumerate(predictions["pose_enc_list"]):
        pose_enc_gt = extri_intri_to_pose_encoding(torch.inverse(c2w_traj_gt)[...,:3,:4], intrs, predictions["images"].shape[-2:])
        T_loss = torch.abs(pose_enc_i[..., :3] - pose_enc_gt[..., :3]).mean()
        R_loss = torch.abs(pose_enc_i[..., 3:7] - pose_enc_gt[..., 3:7]).mean()
        K_loss = torch.abs(pose_enc_i[..., 7:] - pose_enc_gt[..., 7:]).mean()
        pose_loss_i = 25*(T_loss + R_loss) + K_loss
        ln_pose += 0.8**(len(predictions["pose_enc_list"]) - i_t - 1)*(pose_loss_i)
    ln_pose = 5*ln_pose
    if annots["syn_real"] == 1:
        loss = ln_depth_glob + ln_edge + ln_normal + ln_cons + ln_scale_shift + ln_pose + ln_msk_l2
    else:
        loss = ln_cons + ln_pose
        ln_scale_shift = 0*ln_scale_shift
    return {"loss": loss, "ln_depth_glob": ln_depth_glob, "ln_edge": ln_edge, "ln_normal": ln_normal,
                                     "ln_cons": ln_cons, "ln_scale_shift": ln_scale_shift,
                                      "ln_pose": ln_pose, "ln_msk_l2": ln_msk_l2, "norm_scale": norm_rescale}

