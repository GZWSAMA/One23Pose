from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import trimesh
import os
from scipy.spatial import cKDTree
import pytorch_lightning as pl
import random
import open3d as o3d
import utils3d
from trellis.modules.sparse.attention import SparseMultiHeadAttention
from trellis.models.structured_latent_flow import SparseResBlock3d
import trellis.modules.sparse as sp
from .trellis_image_to_3d import *
from scipy.ndimage import binary_dilation
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from trellis.models.sparse_structure_vae import *
from ..modules.spatial import patchify, unpatchify
from easydict import EasyDict as edict
import trellis.models as models
from trellis.modules.transformer import ModulatedPosedTransformerBlock

# import sys
# sys.path.append("wheels/vggt")
# from wheels.vggt.vggt.models.vggt import VGGT
# from wheels.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri, quat_to_mat, mat_to_quat
# from wheels.vggt.vggt.utils.geometry import depth_to_world_coords_points_tensor
# from trellis.modules.transformer import ModulatedMultiViewCond

def get_opencv_from_blender(matrix_world, fov, image_size):
    # convert matrix_world to opencv format extrinsics
    opencv_world_to_cam = matrix_world.inverse()
    opencv_world_to_cam[:, 1, :] *= -1
    opencv_world_to_cam[:, 2, :] *= -1
    R, T = opencv_world_to_cam[:, :3, :3], opencv_world_to_cam[:, :3, 3]
    R, T = R, T
    
    # convert fov to opencv format intrinsics
    opencv_cam_matrix  = torch.diag(torch.tensor([1, 1, 1])).unsqueeze(0).float().repeat(fov.shape[0], 1, 1).to(fov.device)
    focal = 1 / torch.tan(fov / 2)
    opencv_cam_matrix[:, :2, :2] *= focal[:,None,None]
    opencv_cam_matrix[:, :2, -1] += torch.tensor([image_size / 2, image_size / 2]).to(fov.device)[None]
    opencv_cam_matrix[:, :2, :2] *= image_size / 2

    return R, T, opencv_cam_matrix

class TrellisImageToSSPipeline(TrellisImageTo3DPipeline):
    """
    Pipeline for generating sparse structures from images using Trellis.
    Inherits from TrellisImageTo3DPipeline but focuses only on sparse structure generation.
    """
    
    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageToSSPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageToSSPipeline, TrellisImageToSSPipeline).from_pretrained(path)
        new_pipeline = TrellisImageToSSPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        # Initialize the sparse structure sampler
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        # Remove unnecessary models and components since we only need sparse structure
        for key in ['slat_flow_model', 'slat_encoder', 'slat_decoder_gs', 'slat_decoder_mesh', 'slat_decoder_rf']:
            if key in new_pipeline.models:
                del new_pipeline.models[key]
                
        if hasattr(new_pipeline, 'slat_flow_model'):
            del new_pipeline.slat_flow_model
        if hasattr(new_pipeline, 'slat_sampler'):
            del new_pipeline.slat_sampler
        if hasattr(new_pipeline, 'slat_sampler_params'):
            del new_pipeline.slat_sampler_params
        if hasattr(new_pipeline, 'slat_normalization'):
            del new_pipeline.slat_normalization

        # Initialize image conditioning model
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        
        return new_pipeline

    def get_input(self, batch_data):
        """
        Process input batch data for training.

        Args:
            batch_data (dict): Batch data containing images and targets

        Returns:
            tuple: Processed targets, conditioning, and noise
        """
        images = batch_data['source_image']
        cond = self.encode_image(images)
        
        # Randomly zero out conditioning during training for robustness
        if random.random() > 0.5:
            cond = torch.zeros_like(cond)

        targets = batch_data['target_ss_latents'].to(self.device)
        noise = torch.randn_like(targets).to(self.device)

        return targets, cond, noise

    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        preprocess_image: bool = True,
    ) -> torch.Tensor:
        """
        Generate sparse structure from an input image.

        Args:
            image (Image.Image): Input image
            num_samples (int): Number of samples to generate
            seed (int): Random seed
            sparse_structure_sampler_params (dict): Additional sampler parameters
            preprocess_image (bool): Whether to preprocess the input image

        Returns:
            torch.Tensor: Generated sparse structure coordinates
        """
        if preprocess_image:
            image = self.preprocess_image(image)
            
        # Get image conditioning
        cond = self.get_cond([image])
        
        # Sample sparse structure
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        torch.manual_seed(seed)
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        
        # Merge default and custom sampler parameters
        sampler_params = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        
        # Generate occupancy latent
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent to get coordinates
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        
        return coords

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the sparse structure flow model.

        Args:
            x (torch.Tensor): Input tensor
            t (torch.Tensor): Time step
            cond (torch.Tensor): Conditioning tensor

        Returns:
            torch.Tensor: Model output
        """
        return self.sparse_structure_flow_model(x, t, cond)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        params = list(self.sparse_structure_flow_model.parameters())
        # return torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0)

    def p_losses(self, x_0, cond, t, noise):
        """
        Calculate training losses.

        Args:
            x_0 (torch.Tensor): Ground truth tensor
            cond (torch.Tensor): Conditioning tensor  
            t (float): Time step
            noise (torch.Tensor): Noise tensor

        Returns:
            tuple: Loss value and loss dictionary
        """
        x_t, gt_v = self.sparse_structure_sampler._get_model_gt(x_0, t, noise)

        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        pred_v = self(x_t, t, cond)
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        loss_simple = F.mse_loss(pred_v, gt_v)
        # Ignore NaN losses
        valid_loss_simple = loss_simple[~torch.isnan(loss_simple)]
        loss_dict.update({f'{prefix}/loss_simple': valid_loss_simple.mean()})
        loss = valid_loss_simple.mean()
        
        return loss, loss_dict

    def mse_loss(self, pred, target):
        diff = pred - target
        return diff ** 2

class TrellisVGGTLORAToSSPipeline(TrellisImageToSSPipeline):

    def get_input(self, batch_data):

        images = batch_data['ref_image'].to(self.VGGT_dtype)
        b,n,c,h,w = images.shape
        n = random.randint(1, n)
        images = images[:,:n]
        images = F.interpolate(images.reshape(b*n,c,h,w), 518, mode='bilinear')
        coords = F.interpolate(batch_data['source_image'][:,:n].to(self.VGGT_dtype).reshape(b*n,3,h,w), 518, mode='nearest')
        h, w = images.shape[-2:]
        images = images.reshape(b,n,c,h,w)
        coords = coords.reshape(b,n,3,h,w)
        mask = ((coords.permute(0,1,3,4,2)[...,0] == 127/255) & (coords.permute(0,1,3,4,2)[...,1] == 127/255) & (coords.permute(0,1,3,4,2)[...,2] == 127/255))
        mask = ~mask

        aggregated_tokens_list, ps_idx = self.VGGT_model.aggregator(images)
        
        image_cond = self.encode_image(images.reshape(b*n,c,h,w)).reshape(b, n, -1, 1024)[:, :, 5:]
        cond = self.multiview_cond(aggregated_tokens_list, image_cond)
        if random.random() < 0.1:
            cond = torch.zeros_like(cond)

        ss_encoder = self.models['ss_encoder']
        ss_encoder.eval()
        target_coords = batch_data['target_coords'].to(self.device)
        voxel_resolution = 64

        with torch.no_grad():
            ss = torch.zeros(b, voxel_resolution, voxel_resolution, voxel_resolution, dtype=torch.long, device=self.device)
            ss.index_put_((target_coords[:,0], target_coords[:,1], target_coords[:,2], target_coords[:,3]), torch.tensor(1, dtype=ss.dtype, device=ss.device))
            ss = ss.unsqueeze(1).float()
            targets = ss_encoder(ss.to(ss_encoder.dtype), sample_posterior=False).to(self.VGGT_dtype)
        
        # targets = batch_data['target_ss_latents'].to(self.device)
        noise = torch.randn_like(targets).to(self.VGGT_dtype).to(self.device)

        # return targets, cond, noise, pose_enc_gt, pose_enc
        return targets, cond, noise

    @staticmethod
    def from_pretrained(path: str, training=True, resume_path=None) -> "TrellisVGGTLORAToSSPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisVGGTLORAToSSPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.multiview_cond = ModulatedMultiViewCond(
                                        1024,
                                        3072,
                                        num_heads=16,
                                        mlp_ratio=4,
                                        attn_mode='full',
                                        use_checkpoint=False,
                                        use_rope=False,
                                        share_mod=False,
                                        qk_rms_norm=True,
                                        qk_rms_norm_cross=False,
                                    ).to(new_pipeline.device)
        if training:
            # Remove unnecessary models and components since we only need sparse structure
            for key in ['slat_flow_model', 'slat_encoder', 'slat_decoder_gs', 'slat_decoder_mesh', 'slat_decoder_rf']:
                if key in new_pipeline.models:
                    del new_pipeline.models[key]
                    
            if hasattr(new_pipeline, 'slat_flow_model'):
                del new_pipeline.slat_flow_model
            if hasattr(new_pipeline, 'slat_sampler'):
                del new_pipeline.slat_sampler
            if hasattr(new_pipeline, 'slat_sampler_params'):
                del new_pipeline.slat_sampler_params
            if hasattr(new_pipeline, 'slat_normalization'):
                del new_pipeline.slat_normalization

            if resume_path is not None:
                states = torch.load(resume_path)
                if 'state_dict' in states:
                    states = states['state_dict']
                new_pipeline.models['sparse_structure_flow_model'].load_state_dict({k.replace(f"sparse_structure_flow_model.", ""): v for k, v in states.items()}, False)
                new_pipeline.multiview_cond.load_state_dict({k.replace(f"multiview_cond.", ""): v for k, v in states.items()}, False)
        
        new_pipeline.VGGT_dtype = torch.float32
        VGGT_model = VGGT()
        VGGT_model_weight = torch.load("weights/VGGT_weight/model.pt")
        VGGT_model.load_state_dict(VGGT_model_weight)
        new_pipeline.VGGT_model = VGGT_model.to(new_pipeline.device)
        del new_pipeline.VGGT_model.depth_head
        del new_pipeline.VGGT_model.point_head
        del new_pipeline.VGGT_model.track_head
        del new_pipeline.VGGT_model.camera_head
        new_pipeline.VGGT_model.eval()

        new_pipeline.models['ss_encoder'] = models.from_pretrained("weights/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
        new_pipeline.models['ss_encoder'].eval()
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def configure_optimizers(self):
        vggt_lora_params = [p for p in self.VGGT_model.aggregator.parameters() if p.requires_grad]
        trellis_lora_params = [p for p in self.sparse_structure_flow_model.parameters() if p.requires_grad]
        params = list(self.multiview_cond.parameters()) + trellis_lora_params + vggt_lora_params
        opt = FusedAdam(params, 1e-4)
        return opt