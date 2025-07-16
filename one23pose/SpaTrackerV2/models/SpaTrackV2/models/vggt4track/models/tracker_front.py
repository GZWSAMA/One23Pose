# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from models.SpaTrackV2.models.vggt4track.models.aggregator_front import Aggregator
from models.SpaTrackV2.models.vggt4track.heads.camera_head import CameraHead
from models.SpaTrackV2.models.vggt4track.heads.scale_head import ScaleHead
from einops import rearrange
from models.SpaTrackV2.utils.loss import compute_loss
from models.SpaTrackV2.utils.pose_enc import pose_encoding_to_extri_intri
import torch.nn.functional as F

class FrontTracker(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518,
                       patch_size=14, embed_dim=1024, base_model=None, use_checkpoint=True, use_scale_head=False):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        if use_scale_head:
            self.scale_head = ScaleHead(dim_in=2 * embed_dim)
        else:
            self.scale_head = None
        self.base_model = base_model
        self.use_checkpoint = use_checkpoint
        self.intermediate_layers = [4, 11, 17, 23]
        self.residual_proj = nn.ModuleList([nn.Linear(2048, 1024) for _ in range(len(self.intermediate_layers))])
        # init the residual proj
        for i in range(len(self.intermediate_layers)):
            nn.init.xavier_uniform_(self.residual_proj[i].weight)
            nn.init.zeros_(self.residual_proj[i].bias)
        # self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        # self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        # self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(self,
                 images: torch.Tensor,
                 annots = {},
                 **kwargs):
        """
        Forward pass of the FrontTracker model.

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
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, T, C, H, W = images.shape
        images = (images - self.base_model.image_mean) / self.base_model.image_std
        H_14 = H // 14 * 14
        W_14 = W // 14 * 14 
        image_14 = F.interpolate(images.view(B*T, C, H, W), (H_14, W_14), mode="bilinear", align_corners=False, antialias=True).view(B, T, C, H_14, W_14)

        with torch.no_grad():
            features = self.base_model.backbone.get_intermediate_layers(rearrange(image_14, 'b t c h w -> (b t) c h w'), 
                                                                        self.base_model.intermediate_layers, return_class_token=True)
        # aggregate the features with checkpoint
        aggregated_tokens_list, patch_start_idx = self.aggregator(image_14, patch_tokens=features[-1][0])
                
        # enhance the features
        enhanced_features = []
        for layer_i, layer in enumerate(self.intermediate_layers):
            # patch_feat_i = features[layer_i][0] + self.residual_proj[layer_i](aggregated_tokens_list[layer][:,:,patch_start_idx:,:].view(B*T, features[layer_i][0].shape[1], -1))
            patch_feat_i = self.residual_proj[layer_i](aggregated_tokens_list[layer][:,:,patch_start_idx:,:].view(B*T, features[layer_i][0].shape[1], -1))
            enhance_i = (patch_feat_i, features[layer_i][1])
            enhanced_features.append(enhance_i)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            if self.scale_head is not None:
                scale_list = self.scale_head(aggregated_tokens_list)
                predictions["scale"] = scale_list[-1]  # scale of the last iteration
            # Predict points (and mask) with checkpoint
            output = self.base_model.head(enhanced_features, image_14)
            points, mask = output
            
            # Post-process points and mask
            points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)
            points = self.base_model._remap_points(points)     # slightly improves the performance in case of very large output values
            # prepare the predictions
            predictions["images"] = (images * self.base_model.image_std + self.base_model.image_mean)*255.0
            points = F.interpolate(points.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
            predictions["points_map"] = points
            mask = F.interpolate(mask.unsqueeze(1), (H, W), mode="bilinear", align_corners=False, antialias=True).squeeze(1)
            predictions["unc_metric"] = mask
            predictions["pose_enc_list"] = pose_enc_list

        if self.training:
            loss = compute_loss(predictions, annots)
            predictions["loss"] = loss
       
        # rescale the points
        if self.scale_head is not None:
            points_scale = points * predictions["scale"].view(B*T, 1, 1, 2)[..., :1]
            points_scale[..., 2:] += predictions["scale"].view(B*T, 1, 1, 2)[..., 1:]
            predictions["points_map"] = points_scale
    
        predictions["poses_pred"] = torch.eye(4)[None].repeat(predictions["images"].shape[1], 1, 1)[None]
        predictions["poses_pred"][:,:,:3,:4], predictions["intrs"] = pose_encoding_to_extri_intri(predictions["pose_enc_list"][-1],
                                                                                                            predictions["images"].shape[-2:])
        return predictions
