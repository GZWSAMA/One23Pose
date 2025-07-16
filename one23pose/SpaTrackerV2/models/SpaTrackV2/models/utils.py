# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/facebookresearch/PoseDiffusion

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union, List
from einops import rearrange, repeat

import cv2
import numpy as np

# from torchmetrics.functional.regression import pearson_corrcoef
from easydict import EasyDict as edict
from enum import Enum
import torch.utils.data.distributed as dist
from typing import Literal, Union, List, Tuple, Dict
from models.monoD.depth_anything_v2.util.transform import Resize
from models.SpaTrackV2.utils.model_utils import sample_features5d
EPS = 1e-9

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def procrustes_analysis(X0,X1): # [N,3]                                 
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def create_intri_matrix(focal_length, principal_point):
    """
    Creates a intri matrix from focal length and principal point.

    Args:
        focal_length (torch.Tensor): A Bx2 or BxSx2 tensor containing the focal lengths (fx, fy) for each image.
        principal_point (torch.Tensor): A Bx2 or BxSx2 tensor containing the principal point coordinates (cx, cy) for each image.

    Returns:
        torch.Tensor: A Bx3x3 or BxSx3x3 tensor containing the camera matrix for each image.
    """

    if len(focal_length.shape) == 2:
        B = focal_length.shape[0]
        intri_matrix = torch.zeros(B, 3, 3, dtype=focal_length.dtype, device=focal_length.device)
        intri_matrix[:, 0, 0] = focal_length[:, 0]
        intri_matrix[:, 1, 1] = focal_length[:, 1]
        intri_matrix[:, 2, 2] = 1.0
        intri_matrix[:, 0, 2] = principal_point[:, 0]
        intri_matrix[:, 1, 2] = principal_point[:, 1]
    else:
        B, S = focal_length.shape[0], focal_length.shape[1]
        intri_matrix = torch.zeros(B, S, 3, 3, dtype=focal_length.dtype, device=focal_length.device)
        intri_matrix[:, :, 0, 0] = focal_length[:, :, 0]
        intri_matrix[:, :, 1, 1] = focal_length[:, :, 1]
        intri_matrix[:, :, 2, 2] = 1.0
        intri_matrix[:, :, 0, 2] = principal_point[:, :, 0]
        intri_matrix[:, :, 1, 2] = principal_point[:, :, 1]

    return intri_matrix


def closed_form_inverse_OpenCV(se3, R=None, T=None):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.


    | R t |
    | 0 1 |
    -->
    | R^T  -R^T t|
    | 0       1  |
    """
    if R is None:
        R = se3[:, :3, :3]

    if T is None:
        T = se3[:, :3, 3:]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # -R^T t
    top_right = -R_transposed.bmm(T)

    inverted_matrix = torch.eye(4, 4)[None].repeat(len(se3), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def get_EFP(pred_cameras, image_size, B, S, default_focal=False):
    """
    Converting PyTorch3D cameras to extrinsics, intrinsics matrix

    Return extrinsics, intrinsics, focal_length, principal_point
    """
    scale = image_size.min()

    focal_length = pred_cameras.focal_length

    principal_point = torch.zeros_like(focal_length)

    focal_length = focal_length * scale / 2
    principal_point = (image_size[None] - principal_point * scale) / 2

    Rots = pred_cameras.R.clone()
    Trans = pred_cameras.T.clone()

    extrinsics = torch.cat([Rots, Trans[..., None]], dim=-1)

    # reshape
    extrinsics = extrinsics.reshape(B, S, 3, 4)
    focal_length = focal_length.reshape(B, S, 2)
    principal_point = principal_point.reshape(B, S, 2)

    # only one dof focal length
    if default_focal:
        focal_length[:] = scale
    else:
        focal_length = focal_length.mean(dim=-1, keepdim=True).expand(-1, -1, 2)
        focal_length = focal_length.clamp(0.2 * scale, 5 * scale)

    intrinsics = create_intri_matrix(focal_length, principal_point)
    return extrinsics, intrinsics
    
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR_logFL",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=30,
    return_dict=False,
    to_OpenCV=True,
):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """
    pose_encoding_reshaped = pose_encoding.reshape(-1, pose_encoding.shape[-1])  # Reshape to BNxC

    if pose_encoding_type == "absT_quaR_logFL":
        # 3 for absT, 4 for quaR, 2 for absFL
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
        log_focal_length = pose_encoding_reshaped[:, 7:9]
        # log_focal_length_bias was the hyperparameter
        # to ensure the mean of logFL close to 0 during training
        # Now converted back
        focal_length = (log_focal_length + log_focal_length_bias).exp()
        # clamp to avoid weird fl values
        focal_length = torch.clamp(focal_length, 
                                   min=min_focal_length, max=max_focal_length)
    elif pose_encoding_type == "absT_quaR_OneFL":
        # 3 for absT, 4 for quaR, 1 for absFL
        # [absolute translation, quaternion rotation, normalized focal length]
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
        focal_length = pose_encoding_reshaped[:, 7:8]
        focal_length = torch.clamp(focal_length, 
                                   min=min_focal_length, max=max_focal_length)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if to_OpenCV:
        ### From Pytorch3D coordinate to OpenCV coordinate:
        # I hate coordinate conversion
        R = R.clone()
        abs_T = abs_T.clone()
        R[:, :, :2] *= -1
        abs_T[:, :2] *= -1
        R = R.permute(0, 2, 1)

        extrinsics_4x4 = torch.eye(4, 4).to(R.dtype).to(R.device)
        extrinsics_4x4 = extrinsics_4x4[None].repeat(len(R), 1, 1)

        extrinsics_4x4[:, :3, :3] = R.clone()
        extrinsics_4x4[:, :3, 3] = abs_T.clone()

        rel_transform = closed_form_inverse_OpenCV(extrinsics_4x4[0:1])
        rel_transform = rel_transform.expand(len(extrinsics_4x4), -1, -1)

        # relative to the first camera
        # NOTE it is extrinsics_4x4 x rel_transform instead of rel_transform x extrinsics_4x4
        extrinsics_4x4 = torch.bmm(extrinsics_4x4, rel_transform)

        R = extrinsics_4x4[:, :3, :3].clone()
        abs_T = extrinsics_4x4[:, :3, 3].clone()
    
    if return_dict:
        return {"focal_length": focal_length, "R": R, "T": abs_T}

    pred_cameras = PerspectiveCameras(focal_length=focal_length,
                                       R=R, T=abs_T, device=R.device, in_ndc=False)
    return pred_cameras


def camera_to_pose_encoding(
    camera, pose_encoding_type="absT_quaR_logFL", 
    log_focal_length_bias=1.8, min_focal_length=0.1, max_focal_length=30
):
    """
    Inverse to pose_encoding_to_camera
    """
    if pose_encoding_type == "absT_quaR_logFL":
        # Convert rotation matrix to quaternion
        quaternion_R = matrix_to_quaternion(camera.R)

        # Calculate log_focal_length
        log_focal_length = (
            torch.log(torch.clamp(camera.focal_length,
                                   min=min_focal_length, max=max_focal_length))
            - log_focal_length_bias
        )

        # Concatenate to form pose_encoding
        pose_encoding = torch.cat([camera.T, quaternion_R, log_focal_length], dim=-1)

    elif pose_encoding_type == "absT_quaR_OneFL":
        # [absolute translation, quaternion rotation, normalized focal length]
        quaternion_R = matrix_to_quaternion(camera.R)
        focal_length = (torch.clamp(camera.focal_length,
                                     min=min_focal_length,
                                     max=max_focal_length))[..., 0:1]
        pose_encoding = torch.cat([camera.T, quaternion_R, focal_length], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def init_pose_enc(B: int, 
                  S: int, pose_encoding_type: str="absT_quaR_logFL",
                  device: Optional[torch.device]=None):
    """
    Initialize the pose encoding tensor
    args:
        B: batch size
        S: number of frames
        pose_encoding_type: the type of pose encoding
        device: device to put the tensor
    return:
        pose_enc: [B S C]
    """
    if pose_encoding_type == "absT_quaR_logFL":
        C = 9
    elif pose_encoding_type == "absT_quaR_OneFL":
        C = 8
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")
    
    pose_enc = torch.zeros(B, S, C, device=device)
    pose_enc[..., :3] = 0 # absT
    pose_enc[..., 3] = 1 # quaR
    pose_enc[..., 7:] = 1 # logFL
    return pose_enc

def first_pose_enc_norm(pose_enc: torch.Tensor,
                        pose_encoding_type: str="absT_quaR_OneFL",
                        pose_mode: str = "W2C"):
    """
    make sure the poses in on window are normalized by the first frame, where the 
    first frame transformation is the Identity Matrix.
    NOTE: Poses are all W2C
    args:
        pose_enc: [B S C]
    return:
        pose_enc_norm: [B S C]
    """
    B, S, C = pose_enc.shape
    # Pose encoding to Cameras (Pytorch3D coordinate)
    pred_cameras = pose_encoding_to_camera(
        pose_enc, pose_encoding_type=pose_encoding_type,
        to_OpenCV=False
    ) #NOTE: the camera parameters are not in NDC
    
    R = pred_cameras.R    # [B*S, 3, 3]
    T = pred_cameras.T    # [B*S, 3]
    
    Tran_M = torch.cat([R, T.unsqueeze(-1)], dim=-1) # [B*S, 3, 4]
    extra_ = torch.tensor([[[0, 0, 0, 1]]],
                          device=Tran_M.device).expand(Tran_M.shape[0], -1, -1)
    Tran_M = torch.cat([Tran_M, extra_
                        ], dim=1)
    Tran_M = rearrange(Tran_M, '(b s) c d -> b s c d', b=B)
    
    # Take the first frame as the base of world coordinate
    if pose_mode == "C2W":
        Tran_M_new = (Tran_M[:,:1,...].inverse())@Tran_M
    elif pose_mode == "W2C":
        Tran_M_new = Tran_M@(Tran_M[:,:1,...].inverse())
    
    Tran_M_new = rearrange(Tran_M_new, 'b s c d -> (b s) c d')

    R_ = Tran_M_new[:, :3, :3]
    T_ = Tran_M_new[:, :3, 3]

    # Cameras to Pose encoding
    pred_cameras.R = R_
    pred_cameras.T = T_
    pose_enc_norm = camera_to_pose_encoding(pred_cameras,
                                             pose_encoding_type=pose_encoding_type)  
    pose_enc_norm = rearrange(pose_enc_norm, '(b s) c -> b s c', b=B)
    return pose_enc_norm

def first_pose_enc_denorm(
                        pose_enc: torch.Tensor,
                        pose_enc_1st: torch.Tensor,
                        pose_encoding_type: str="absT_quaR_OneFL",
                        pose_mode: str = "W2C"):
    """
    make sure the poses in on window are de-normalized by the first frame, where the 
    first frame transformation is the Identity Matrix.
    args:
        pose_enc: [B S C]
        pose_enc_1st: [B 1 C]
    return:
        pose_enc_denorm: [B S C]
    """
    B, S, C = pose_enc.shape
    pose_enc_all = torch.cat([pose_enc_1st, pose_enc], dim=1)

    # Pose encoding to Cameras (Pytorch3D coordinate)
    pred_cameras = pose_encoding_to_camera(
        pose_enc_all, pose_encoding_type=pose_encoding_type,
        to_OpenCV=False
    ) #NOTE: the camera parameters are not in NDC
    R = pred_cameras.R    # [B*(1+S), 3, 3]
    T = pred_cameras.T    # [B*(1+S), 3]
    
    Tran_M = torch.cat([R, T.unsqueeze(-1)], dim=-1) # [B*(1+S), 3, 4]
    extra_ = torch.tensor([[[0, 0, 0, 1]]],
                          device=Tran_M.device).expand(Tran_M.shape[0], -1, -1)
    Tran_M = torch.cat([Tran_M, extra_
                        ], dim=1)
    Tran_M_new = rearrange(Tran_M, '(b s) c d -> b s c d', b=B)[:, 1:]
    Tran_M_1st = rearrange(Tran_M, '(b s) c d -> b s c d', b=B)[:,:1]

    if pose_mode == "C2W":
        Tran_M_new = Tran_M_1st@Tran_M_new
    elif pose_mode == "W2C":
        Tran_M_new = Tran_M_new@Tran_M_1st
    
    Tran_M_new_ = torch.cat([Tran_M_1st, Tran_M_new], dim=1)
    R_ = Tran_M_new_[..., :3, :3].view(-1, 3, 3)
    T_ = Tran_M_new_[..., :3, 3].view(-1, 3)

    # Cameras to Pose encoding
    pred_cameras.R = R_
    pred_cameras.T = T_

    # Cameras to Pose encoding
    pose_enc_denorm = camera_to_pose_encoding(pred_cameras,
                                             pose_encoding_type=pose_encoding_type)  
    pose_enc_denorm = rearrange(pose_enc_denorm, '(b s) c -> b s c', b=B)
    return pose_enc_denorm[:, 1:]

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def normalize_prediction_robust(target, mask, Bs):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum).to(target.dtype)
    s = torch.ones_like(ssum).to(target.dtype)
    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = rearrange(target, '(b c) h w -> b c h w', b=Bs)
    m_vid = rearrange(m, '(b c) -> b c 1 1', b=Bs)   #.mean(dim=1, keepdim=True)
    mask = rearrange(mask, '(b c) h w -> b c h w', b=Bs)

    target = target - m_vid

    sq = torch.sum(mask * target.abs(), (2, 3))
    sq = rearrange(sq, 'b c -> (b c)')
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)
    s_vid = rearrange(s, '(b c) -> b c 1 1', b=Bs)  #.mean(dim=1, keepdim=True)
    target = target / s_vid
    target = rearrange(target, 'b c h w -> (b c) h w', b=Bs)

    return target, m_vid, s_vid

def normalize_video_robust(target, mask, Bs):

    vid_valid = target[mask]
    # downsample to 1/20
    with torch.no_grad():
        vid_valid = vid_valid[torch.randperm(vid_valid.shape[0], device='cuda')[:vid_valid.shape[0]//5]]
        t_2, t_98 = torch.quantile(vid_valid, 0.02), torch.quantile(vid_valid, 0.98)
    # normalize
    target = (target - t_2) / (t_98 - t_2)*2 - 1
    return target, t_2, t_98

def video_loss(prediction, target, mask, Bs):
    # median norm
    prediction_nm, a_norm, b_norm = normalize_video_robust(prediction, mask, Bs)
    target_nm, a_norm_gt, b_norm_gt = normalize_video_robust(target.float(), mask, Bs)
    depth_loss = nn.functional.l1_loss(prediction_nm[mask], target_nm[mask])
    # rel depth 2 metric --> (pred - a')/(b'-a')*(b-a) + a
    scale = (b_norm_gt - a_norm_gt) / (b_norm - a_norm)
    shift = a_norm_gt - a_norm*scale
    return depth_loss, scale, shift, prediction_nm, target_nm

def median_loss(prediction, target, mask, Bs):
    # median norm
    prediction_nm, a_norm, b_norm = normalize_prediction_robust(prediction, mask, Bs)
    target_nm, a_norm_gt, b_norm_gt = normalize_prediction_robust(target.float(), mask, Bs)
    depth_loss = nn.functional.l1_loss(prediction_nm[mask], target_nm[mask])
    scale = b_norm_gt/b_norm
    shift = a_norm_gt - a_norm*scale 
    return depth_loss, scale, shift, prediction_nm, target_nm

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, Bs,
                 interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."
        

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        a_norm = scale.view(Bs, -1, 1, 1).mean(dim=1, keepdim=True)
        b_norm = shift.view(Bs, -1, 1, 1).mean(dim=1, keepdim=True)
        prediction = rearrange(prediction, '(b c) h w -> b c h w', b=Bs)
        target = rearrange(target, '(b c) h w -> b c h w', b=Bs)
        mask = rearrange(mask, '(b c) h w -> b c h w', b=Bs)
        scaled_prediction = a_norm * prediction + b_norm
        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss, a_norm, b_norm
        return loss, a_norm, b_norm

ScaleAndShiftInvariantLoss_fn = ScaleAndShiftInvariantLoss()

class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)
            l1_ln, a_nm, b_nm = ScaleAndShiftInvariantLoss_fn(prediction[:, ::step, ::step], 
                                                   target[:, ::step, ::step], mask[:, ::step, ::step], 1)
            total += l1_ln
            a_nm = a_nm.squeeze().detach()  # [B, 1, 1]
            b_nm = b_nm.squeeze().detach()  # [B, 1, 1]
            total += 2*gradient_loss(a_nm*prediction[:, ::step, ::step]+b_nm, target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

Grad_fn = GradientLoss()

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)
  
def loss_fn(
        poses_preds: List[torch.Tensor],
        poses_pred_all: List[torch.Tensor],
        poses_gt: torch.Tensor,
        inv_depth_preds: List[torch.Tensor],
        inv_depth_raw: List[torch.Tensor],
        depths_gt: torch.Tensor,
        S: int = 16,
        gamma: float = 0.8,
        logger=None,
        logger_tf=None,
        global_step=0,
        ):
    """
    Args:
        poses_preds: list of predicted poses
        poses_gt: ground truth poses
        inv_depth_preds: list of predicted inverse depth maps
        depths_gt: ground truth depth maps
        S: length of sliding window
    """
    B, T, _, H, W = depths_gt.shape

    loss_total = 0
    for i in range(len(poses_preds)):
        poses_preds_i = poses_preds[i][0]
        poses_unc_i = poses_preds[i][1]
        poses_gt_i = poses_gt[:, i*S//2:i*S//2+S,:]
        poses_gt_i_norm = first_pose_enc_norm(poses_gt_i,
                                               pose_encoding_type="absT_quaR_OneFL")
        pose_loss = 0.0
        for idx, poses_preds_ij in enumerate(poses_preds_i):
            i_weight = gamma ** (len(poses_preds_i) - idx - 1)
            if logger is not None:
                if poses_preds_ij.max()>5e1:
                    logger.info(f"pose_pred_max_and_mean: {poses_preds_ij.max(), poses_preds_ij.mean()}")
            
            trans_loss = (poses_preds_ij[...,:3] - poses_gt_i_norm[...,:3]).abs().sum(dim=-1).mean()
            rot_loss = (poses_preds_ij[...,3:7] - poses_gt_i_norm[...,3:7]).abs().sum(dim=-1).mean()
            focal_loss = (poses_preds_ij[...,7:] - poses_gt_i_norm[...,7:]).abs().sum(dim=-1).mean()
            if torch.isnan((trans_loss + rot_loss + focal_loss)).any():
                pose_loss += 0
            else:
                pose_loss += i_weight*(trans_loss + rot_loss + focal_loss)
            if (logger_tf is not None)&(i==len(poses_preds)-1):
                logger_tf.add_scalar(f"loss@pose/trans_iter{idx}",
                                            trans_loss, global_step=global_step)
                logger_tf.add_scalar(f"loss@pose/rot_iter{idx}",
                                            rot_loss, global_step=global_step)
                logger_tf.add_scalar(f"loss@pose/focal_iter{idx}",
                                            focal_loss, global_step=global_step)   
        # compute the uncertainty loss
        with torch.no_grad():
            pose_loss_dist = (poses_preds_ij-poses_gt_i_norm).detach().abs()
            pose_loss_std = 3*pose_loss_dist.view(-1,8).std(dim=0)[None,None,:]
            gt_dist =  F.relu(pose_loss_std - pose_loss_dist) / (pose_loss_std + 1e-3)
        unc_loss = (gt_dist - poses_unc_i).abs().mean()
        if (logger_tf is not None)&(i==len(poses_preds)-1):
            logger_tf.add_scalar(f"loss@uncertainty/unc",
                                    unc_loss,
                                    global_step=global_step)
        # if logger is not None:
        #     logger.info(f"pose_loss: {pose_loss}, unc_loss: {unc_loss}")   
        # total loss
        loss_total += 0.1*unc_loss + 2*pose_loss 

    poses_gt_norm = poses_gt
    pose_all_loss = 0.0
    prev_loss = None
    for idx, poses_preds_all_j in enumerate(poses_pred_all):
        i_weight = gamma ** (len(poses_pred_all) - idx - 1)
        trans_loss = (poses_preds_all_j[...,:3] - poses_gt_norm[...,:3]).abs().sum(dim=-1).mean()
        rot_loss = (poses_preds_all_j[...,3:7] - poses_gt_norm[...,3:7]).abs().sum(dim=-1).mean()
        focal_loss = (poses_preds_all_j[...,7:] - poses_gt_norm[...,7:]).abs().sum(dim=-1).mean()
        if (logger_tf is not None):
            if prev_loss is None:
                prev_loss = (trans_loss + rot_loss + focal_loss)
            else:
                des_loss = (trans_loss + rot_loss + focal_loss) - prev_loss
                prev_loss = trans_loss + rot_loss + focal_loss 
                logger_tf.add_scalar(f"loss@global_pose/des_iter{idx}",
                                        des_loss, global_step=global_step)
            logger_tf.add_scalar(f"loss@global_pose/trans_iter{idx}",
                                        trans_loss, global_step=global_step)
            logger_tf.add_scalar(f"loss@global_pose/rot_iter{idx}",
                                        rot_loss, global_step=global_step)
            logger_tf.add_scalar(f"loss@global_pose/focal_iter{idx}",
                                        focal_loss, global_step=global_step) 
        if torch.isnan((trans_loss + rot_loss + focal_loss)).any():
            pose_all_loss += 0
        else:
            pose_all_loss += i_weight*(trans_loss + rot_loss + focal_loss)
            
    # if logger is not None:
    #     logger.info(f"global_pose_loss: {pose_all_loss}")  

    # compute the depth loss
    if inv_depth_preds[0] is not None:
        depths_gt = depths_gt[:,:,0]
        msk = depths_gt > 5e-2
        inv_gt = 1.0 / (depths_gt.clamp(1e-3, 1e16))  
        inv_gt_reshp = rearrange(inv_gt, 'b t h w -> (b t) h w')
        inv_depth_preds_reshp = rearrange(inv_depth_preds[0], 'b t h w -> (b t) h w')
        inv_raw_reshp = rearrange(inv_depth_raw[0], 'b t h w -> (b t) h w')
        msk_reshp = rearrange(msk, 'b t h w -> (b t) h w')
        huber_loss = ScaleAndShiftInvariantLoss_fn(inv_depth_preds_reshp, inv_gt_reshp, msk_reshp)
        huber_loss_raw = ScaleAndShiftInvariantLoss_fn(inv_raw_reshp, inv_gt_reshp, msk_reshp)
        # huber_loss = (inv_depth_preds[0][msk]-inv_gt[msk]).abs().mean()
        # cal perason loss
        perason_loss = 0
        # for i in range(B):
        #     perason_loss += (1 - pearson_corrcoef(inv_depth_preds[0].view(B*T,-1), inv_gt.view(B*T,-1))).mean()
        # perason_loss = perason_loss/B
        if torch.isnan(huber_loss).any():
            huber_loss = 0
        depth_loss = huber_loss + perason_loss
        if (logger_tf is not None)&(i==len(poses_preds)-1):
            logger_tf.add_scalar(f"loss@depth/huber_iter{idx}",
                                        depth_loss,
                                        global_step=global_step)
        # if logger is not None:
        #         logger.info(f"opt_depth: {huber_loss_raw - huber_loss}")   
    else:
        depth_loss = 0.0

    
    loss_total = loss_total/(len(poses_preds)) + 20*depth_loss + pose_all_loss

    return loss_total, (huber_loss_raw - huber_loss)


def vis_depth(x: torch.tensor,
              logger_tf = None, title: str = "depth", step: int = 0):
    """
    args:
        x: H W
    """
    assert len(x.shape) == 2

    depth_map_normalized = cv2.normalize(x.cpu().numpy(), 
                                        None, 0, 255, cv2.NORM_MINMAX)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8),
                                        cv2.COLORMAP_JET)
    depth_map_tensor = torch.from_numpy(depth_map_colored).permute(2, 0, 1).unsqueeze(0)
    if logger_tf is not None:
        logger_tf.add_image(title, depth_map_tensor[0], step)
    else:
        return depth_map_tensor

def vis_pcd(
        rgbs: torch.Tensor,
        R: torch.Tensor,
        T: torch.Tensor,
        xy_depth: torch.Tensor,
        focal_length: torch.Tensor,
        pick_idx: List = [0]
        ):
    """
    args:
        rgbs: [S C H W]
        R: [S 3 3]
        T: [S 3]
        xy_depth: [S H W 3]
        focal_length: [S]
        pick_idx: list of the index to pick
    """
    S, C, H, W = rgbs.shape

    rgbs_pick = rgbs[pick_idx]
    R_pick = R[pick_idx]
    T_pick = T[pick_idx]
    xy_depth_pick = xy_depth[pick_idx]
    focal_length_pick = focal_length[pick_idx]
    pcd_world = depth2pcd(xy_depth_pick.clone(),
                        focal_length_pick, R_pick.clone(), T_pick.clone(),
                        device=xy_depth.device, H=H, W=W)
    pcd_world = pcd_world.permute(0, 2, 1)                   #[...,[1,0,2]]
    mask = pcd_world.reshape(-1,3)[:,2] < 20
    rgb_world = rgbs_pick.view(len(pick_idx), 3, -1).permute(0, 2, 1)
    pcl = Pointclouds(points=[pcd_world.reshape(-1,3)[mask]],
                    features=[rgb_world.reshape(-1,3)[mask]/255])
    return pcl

def vis_result(rgbs, poses_pred, poses_gt,
                depth_gt, depth_pred, iter_num=0, 
                vis=None, logger_tf=None, cfg=None):
    """
    Args:
        rgbs: [S C H W]
        depths_gt: [S C H W]
        poses_gt: [S C]
        poses_pred: [S C]
        depth_pred: [S H W]
    """
    assert len(rgbs.shape) == 4, "only support one sequence, T 3 H W of rbg"

    if vis is None:
        return
    S, _, H, W = depth_gt.shape
    # get the xy 
    yx = torch.meshgrid(torch.arange(H).to(depth_pred.device),
                        torch.arange(W).to(depth_pred.device),indexing='ij')
    xy = torch.stack(yx[::-1], dim=0).float().to(depth_pred.device)
    xy_norm = (xy / torch.tensor([W, H],
                                    device=depth_pred.device).view(2, 1, 1) - 0.5)*2
    xy = xy[None].repeat(S, 1, 1, 1)
    xy_depth = torch.cat([xy, depth_pred[:,None]], dim=1).permute(0, 2, 3, 1)
    xy_depth_gt = torch.cat([xy, depth_gt], dim=1).permute(0, 2, 3, 1)
    # get the focal length
    focal_length = poses_gt[:,-1]*max(H, W)

    # vis the camera poses
    poses_gt_vis = pose_encoding_to_camera(poses_gt,
                                            pose_encoding_type="absT_quaR_OneFL",to_OpenCV=False)
    poses_pred_vis = pose_encoding_to_camera(poses_pred,
                                              pose_encoding_type="absT_quaR_OneFL",to_OpenCV=False)
    
    R_gt = poses_gt_vis.R.float()
    R_pred = poses_pred_vis.R.float()
    T_gt = poses_gt_vis.T.float()
    T_pred = poses_pred_vis.T.float()
    # C2W poses
    R_gt_c2w = R_gt.permute(0,2,1)
    T_gt_c2w = (-R_gt_c2w @ T_gt[:, :, None]).squeeze(-1)
    R_pred_c2w = R_pred.permute(0,2,1)
    T_pred_c2w = (-R_pred_c2w @ T_pred[:, :, None]).squeeze(-1)
    with torch.cuda.amp.autocast(enabled=False):
        pick_idx = torch.randperm(S)[:min(24, S)] 
        # pick_idx = [1]
        #NOTE: very strange that the camera need C2W Rotation and W2C translation as input
        poses_gt_vis = PerspectiveCamerasVisual(
            R=R_gt_c2w[pick_idx], T=T_gt[pick_idx],
            device=poses_gt_vis.device, image_size=((H, W),)
        )
        poses_pred_vis = PerspectiveCamerasVisual(
            R=R_pred_c2w[pick_idx], T=T_pred[pick_idx],
            device=poses_pred_vis.device
        )
        visual_dict = {"scenes": {"cameras": poses_pred_vis, "cameras_gt": poses_gt_vis}}
        env_name = f"train_visualize_iter_{iter_num:05d}"
        print(f"Visualizing the scene by visdom at env: {env_name}")
        # visualize the depth map
        vis_depth(depth_pred[0].detach(), logger_tf, title="vis/depth_pred",step=iter_num)
        msk = depth_pred[0] > 1e-3
        vis_depth(depth_gt[0,0].detach(), logger_tf, title="vis/depth_gt",step=iter_num)
        depth_res = (depth_gt[0,0] - depth_pred[0]).abs()
        vis_depth(depth_res.detach(), logger_tf, title="vis/depth_res",step=iter_num)
        # visualize the point cloud
        if cfg.debug.vis_pcd:
            visual_dict["scenes"]["points_gt"] = vis_pcd(rgbs, R_gt, T_gt,
                                                        xy_depth_gt, focal_length, pick_idx)
        else:
            visual_dict["scenes"]["points_pred"] = vis_pcd(rgbs, R_pred, T_pred,
                                                            xy_depth, focal_length, pick_idx)
        # visualize in visdom
        fig = plot_scene(visual_dict, camera_scale=0.05)
        vis.plotlyplot(fig, env=env_name, win="3D")
        vis.save([env_name])
        
    return
    
def depth2pcd(
        xy_depth: torch.Tensor,
        focal_length: torch.Tensor,
        R: torch.Tensor,
        T: torch.Tensor,
        device: torch.device = None,
        H: int = 518,
        W: int = 518
    ):
    """
    args:
        xy_depth: [S H W 3]
        focal_length: [S]
        R: [S 3 3]   W2C
        T: [S 3]     W2C
    return:
        xyz: [S 3 (H W)]
    """
    S, H, W, _ = xy_depth.shape
    # get the intrinsic
    K = torch.eye(3, device=device)[None].repeat(len(focal_length), 1, 1).to(device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 0, 2] = 0.5 * W
    K[:, 1, 2] = 0.5 * H
    K_inv = K.inverse()
    # xyz
    xyz = xy_depth.view(S, -1, 3).permute(0, 2, 1) # S 3 (H W)
    depth = xyz[:, 2:].clone() # S (H W) 1 
    xyz[:, 2] = 1
    xyz = K_inv @ xyz # S 3 (H W)
    xyz = xyz * depth
    # to world coordinate
    xyz = R.permute(0,2,1) @ (xyz - T[:, :, None])

    return xyz


def pose_enc2mat(poses_pred, 
                 H_resize, W_resize, resolution=336):
    """
    This function convert the pose encoding into `intrinsic` and `extrinsic`

    Args:
        poses_pred: B T 8
    Return: 
        Intrinsic B T 3 3
        Extrinsic B T 4 4
    """
    B, T, _ = poses_pred.shape
    focal_pred = poses_pred[:, :, -1].clone()
    pos_quat_preds = poses_pred[:, :, :7].clone()   
    pos_quat_preds = pos_quat_preds.view(B*T, -1)  
    # get extrinsic 
    c2w_rot = quaternion_to_matrix(pos_quat_preds[:, 3:])
    c2w_tran = pos_quat_preds[:, :3]
    c2w_traj = torch.eye(4)[None].repeat(B*T, 1, 1).to(poses_pred.device)
    c2w_traj[:, :3, :3], c2w_traj[:, :3, 3] = c2w_rot, c2w_tran
    c2w_traj = c2w_traj.view(B, T, 4, 4)
    # get intrinsic
    fxs, fys = focal_pred*resolution, focal_pred*resolution 
    intrs = torch.eye(3).to(c2w_traj.device).to(c2w_traj.dtype)[None, None].repeat(B, T, 1, 1)
    intrs[:,:,0,0], intrs[:,:,1,1] = fxs, fys
    intrs[:,:,0,2], intrs[:,:,1,2] = W_resize/2, H_resize/2

    return intrs, c2w_traj

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
    
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def meshgrid2d(B, Y, X, stack=False, norm=False, device="cuda"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
    
def get_points_on_a_grid(grid_size, interp_shape,
                          grid_center=(0, 0), device="cuda"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, 
                             interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy

def normalize_rgb(x,input_size=224, 
                resize_mode: Literal['resize', 'padding'] = 'resize',
                if_da=False):
        """
        normalize the image for depth anything input
        
        args:
            x: the input images  [B T C H W]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x) / 255.0
        elif isinstance(x, torch.Tensor):
            x = x / 255.0
        B, T, C, H, W = x.shape     
        x = x.view(B * T, C, H, W)  
        Resizer = Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
            )     
        if resize_mode == 'padding':
            # zero padding to make the input size to be multiple of 14
            if H > W:
                H_scale = input_size
                W_scale = W * input_size // H
            else:
                W_scale = input_size
                H_scale = H * input_size // W
            # resize the image
            x = F.interpolate(x, size=(H_scale, W_scale),
                                mode='bilinear', align_corners=False)
            # central padding the image
            padding_x = (input_size - W_scale) // 2
            padding_y = (input_size - H_scale) // 2
            extra_x = (input_size - W_scale) % 2
            extra_y = (input_size - H_scale) % 2
            x = F.pad(x, (padding_x, padding_x+extra_x,
                        padding_y, padding_y+extra_y), value=0.)
        elif resize_mode == 'resize':
            H_scale, W_scale = Resizer.get_size(H, W)
            x = F.interpolate(x, size=(int(H_scale), int(W_scale)),
                                    mode='bicubic', align_corners=True)
        # get the mean and std
        __mean__ = torch.tensor([0.485, 
                                 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        __std__ = torch.tensor([0.229,
                                 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        # normalize the image
        if if_da:
            x = (x - __mean__) / __std__
        else:
            x = x 
        return x.view(B, T, C, x.shape[-2], x.shape[-1])

def get_track_points(H, W, T, device, size=100, support_frame=0,
                                query_size=768, unc_metric=None, mode="mixed"):
    """
    This function is used to get the points on the grid
    args:
        H: the height of the grid.
        W: the width of the grid.
        T: the number of frames.
        device: the device of the points.
        size: the size of the grid.
    """
    grid_pts = get_points_on_a_grid(size, (H, W), device=device)
    grid_pts = grid_pts.round()
    if mode == "incremental":
        queries = torch.cat(
                [torch.randint_like(grid_pts[:, :, :1], T), grid_pts],
                dim=2,
            )
    elif mode == "first":
        queries_first = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts],
                dim=2,
            )
        queries_support = torch.cat(
                [torch.randint_like(grid_pts[:, :, :1],  T), grid_pts],
                dim=2,
            )
        queries = torch.cat([queries_first, queries_support, queries_support], dim=1)
    elif mode == "mixed":
        queries = torch.cat(
                [torch.randint_like(grid_pts[:, :, :1], T), grid_pts],
                dim=2,
            )
        queries_first = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * support_frame, grid_pts],
                dim=2,
            )
        queries = torch.cat([queries_first, queries, queries], dim=1)
    if unc_metric is not None:
        # filter the points with high uncertainty
        sample_unc = sample_features5d(unc_metric[None], queries[:,None]).squeeze()
        if ((sample_unc>0.5).sum() < 20):
            queries = queries
        else:
            queries = queries[:,sample_unc>0.5,:]
    idx_ = torch.randperm(queries.shape[1], device=device)[:query_size]
    queries = queries[:, idx_]
    return queries