# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from models.SpaTrackV2.models.vggt4track.models.aggregator import Aggregator
from models.SpaTrackV2.models.vggt4track.heads.camera_head import CameraHead
from models.SpaTrackV2.models.vggt4track.heads.dpt_head import DPTHead
from models.SpaTrackV2.models.vggt4track.heads.track_head import TrackHead
from models.SpaTrackV2.models.vggt4track.utils.loss import compute_loss
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import depth_to_points_colmap, get_nth_visible_time_index
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from einops import rearrange
import torch.nn.functional as F

class VGGT4Track(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid")

    def forward(
        self,
        images: torch.Tensor,
        annots = {},
        fx_prev = None,
        fy_prev = None,
        **kwargs):
        """
        Forward pass of the VGGT4Track model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        B, T, C, H, W = images.shape
        images_proc = preprocess_image(images.view(B*T, C, H, W).clone())
        images_proc = rearrange(images_proc, '(b t) c h w -> b t c h w', b=B, t=T)
        _, _, _, H_proc, W_proc = images_proc.shape

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images_proc)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images_proc, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["unc_metric"] = depth_conf.view(B*T, H_proc, W_proc)

        predictions["images"] = (images)*255.0
                                                         
        # output the camera pose
        predictions["poses_pred"] = torch.eye(4)[None].repeat(T, 1, 1)[None]
        predictions["poses_pred"][:,:,:3,:4], predictions["intrs"] = pose_encoding_to_extri_intri(predictions["pose_enc_list"][-1],
                                                                                                                     images_proc.shape[-2:])
        predictions["poses_pred"] = torch.inverse(predictions["poses_pred"])

        if fx_prev is not None:
            scale_x = torch.from_numpy(fx_prev).to(predictions["intrs"].device) / predictions["intrs"][0, :fx_prev.shape[0], 0, 0]
            scale_x = scale_x.mean() * W_proc / W 
            predictions["intrs"][:, :, 0, 0] *= scale_x
        if fy_prev is not None:
            scale_y = torch.from_numpy(fy_prev).to(predictions["intrs"].device) / predictions["intrs"][0, :fy_prev.shape[0], 1, 1]
            scale_y = scale_y.mean() * H_proc / H
            predictions["intrs"][:, :, 1, 1] *= scale_y

        # get the points map
        points_map = depth_to_points_colmap(depth.view(B*T, H_proc, W_proc), predictions["intrs"].view(B*T, 3, 3))
        predictions["points_map"] = points_map
        #NOTE: resize back
        predictions["points_map"] = F.interpolate(points_map.permute(0,3,1,2),
                                                         size=(H, W), mode='bilinear', align_corners=True).permute(0,2,3,1)

        predictions["unc_metric"] = F.interpolate(predictions["unc_metric"][:,None],
                                                         size=(H, W), mode='bilinear', align_corners=True)[:,0]
        predictions["intrs"][..., :1, :] *= W/W_proc 
        predictions["intrs"][..., 1:2, :] *= H/H_proc 

        if self.training:
            loss = compute_loss(predictions, annots)
            predictions["loss"] = loss
                                                                                   
        return predictions
