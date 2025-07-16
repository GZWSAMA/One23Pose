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
from trellis.models.structured_latent_flow import SparseResBlock3d, SparseResBlock3dwoT
import trellis.modules.sparse as sp
from trellis.models.sparse_structure_vae import *
from ..modules.spatial import patchify, unpatchify
from deepspeed.ops.adam import FusedAdam
from trellis.modules.utils import convert_module_to_bf16, convert_module_to_f16, convert_module_to_f32
from contextlib import contextmanager
import trellis.models as models
from trellis.modules.transformer import ModulatedPosedTransformerBlock

import sys
# sys.path.append("wheels/vggt")
# from wheels.vggt.vggt.models.vggt import VGGT
from typing import *
from scipy.spatial.transform import Rotation
from trellis.modules.transformer import ModulatedMultiViewCond

def export_point_cloud(xyz, color):
    # Convert tensors to numpy arrays if needed
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()
    
    color = (color * 255).astype(np.uint8)

    # Create point cloud using trimesh
    point_cloud = trimesh.PointCloud(vertices=xyz, colors=color)
    
    return point_cloud

def normalize_trimesh(mesh):
    # Calculate the mesh centroid and bounding box extents
    centroid = mesh.centroid
    # Determine the scale based on the largest extent to fit into unit cube
    # Normalizing: Center and scale the vertices
    mesh.vertices -= centroid

    extents = mesh.extents
    scale = max(extents)
    mesh.vertices /= scale

    return mesh

def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

from scipy.ndimage import binary_dilation
def voxelize_trimesh(mesh, resolution=(64, 64, 64), stride=4):
    """
    Voxelize a given trimesh object with the specified resolution, incorporating 4x anti-aliasing.
    First voxelizes at a 4x resolution and then downsamples to the target resolution.

    Args:
        mesh (trimesh.Trimesh): The input trimesh object to be voxelized.
        resolution (tuple): The voxel grid resolution as (x, y, z). Default is (64, 64, 64).

    Returns:
        np.ndarray: A boolean numpy array representing the voxel grid where True indicates
                    the presence of the mesh in that voxel and False otherwise.
    """
    target_density = max(resolution)
    target_edge_length = 1.0 / target_density
    max_edge_for_subdivision = target_edge_length / 2  

    # Calculate the higher resolution for 4x anti-aliasing
    anti_aliasing_density = target_density * stride
    anti_aliasing_edge_length = 1.0 / anti_aliasing_density
    anti_aliasing_max_edge_for_subdivision = anti_aliasing_edge_length / 2  

    # Get the vertices and faces of the mesh
    vertices = mesh.vertices
    faces = mesh.faces

    # Subdivide the mesh for the higher resolution voxelization
    try:
        new_vertices, new_faces = trimesh.remesh.subdivide_to_size(
            vertices, faces, anti_aliasing_max_edge_for_subdivision
        )
        subdivided_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    except Exception as e:
        print(f"Unexpected error during mesh subdivision for anti-aliasing: {e}")
        raise

    # Voxelize the subdivided mesh at the higher resolution
    try:
        high_res_voxel_grid = subdivided_mesh.voxelized(
            pitch=anti_aliasing_edge_length, method="binvox", exact=True
        )
    except:
        print("Voxelization using 'binvox' method failed for anti-aliasing")
        high_res_voxel_grid = subdivided_mesh.voxelized(pitch=anti_aliasing_edge_length)
        print("Falling back to default voxelization method for anti-aliasing.")
    high_res_boolean_array = high_res_voxel_grid.matrix.astype(bool)

    x_stride, y_stride, z_stride = [int(anti_aliasing_density / target_density)] * 3
    downsampled_shape = (
        high_res_boolean_array.shape[0] // x_stride,
        high_res_boolean_array.shape[1] // y_stride,
        high_res_boolean_array.shape[2] // z_stride
    )
    downsampled_array = np.zeros(downsampled_shape, dtype=bool)

    # Use NumPy's strided tricks to efficiently access sub-cubes for downsampling
    shape = (downsampled_shape[0], downsampled_shape[1], downsampled_shape[2], x_stride, y_stride, z_stride)
    strides = (x_stride * high_res_boolean_array.strides[0],
               y_stride * high_res_boolean_array.strides[1],
               z_stride * high_res_boolean_array.strides[2],
               high_res_boolean_array.strides[0],
               high_res_boolean_array.strides[1],
               high_res_boolean_array.strides[2])
    sub_cubes = np.lib.stride_tricks.as_strided(high_res_boolean_array, shape=shape, strides=strides)
    downsampled_array = np.any(sub_cubes, axis=(3, 4, 5))

    return downsampled_array

def get_occupied_coordinates(voxel_grid):
    # Find the indices of occupied voxels
    occupied_indices = np.argwhere(voxel_grid)
    
    coords = torch.tensor(occupied_indices, dtype=torch.int8)  # Use float for scaling operations
    
    # Add a leading dimension for batch size or any additional data associations
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords + 1], dim=1)

    # Move to GPU if required
    coords = coords.to('cuda:0')
    
    return coords

from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    default_image_resolution = 518
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        try:
            dinov2_model = torch.hub.load(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main'), name, source='local',pretrained=True)
        except:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image, resolution=518, no_background=True) -> Image.Image:
        """
        Preprocess the input image using BiRefNet for background removal.
        Includes padding to maintain aspect ratio when resizing to 518x518.
        """
        # if has alpha channel, use it directly
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, -1]
            if not np.all(alpha == 255):
                has_alpha = True
        
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            
            # Load BiRefNet model if not already loaded
            if getattr(self, 'birefnet_model', None) is None:
                self._lazy_load_birefnet()
            
            # Get mask using BiRefNet
            mask = self._get_birefnet_mask(input)
            
            # Convert input to RGBA and apply mask
            input_rgba = input.convert('RGBA')
            input_array = np.array(input_rgba)
            input_array[:, :, 3] = mask * 255  # Apply mask to alpha channel
            output = Image.fromarray(input_array)

        # Process the output image
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        
        # Find bounding box of non-transparent pixels
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) == 0:  # Handle case where no foreground is detected
            return input.convert('RGB')
        
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        
        # Calculate and apply crop bbox
        if not no_background:
            height, width = alpha.shape
            if height > width:
                center[0] = width / 2
                if center[1] < width / 2:
                    center[1] = width / 2
                elif center[1] > height - width / 2:
                    center[1] = height - width / 2
            else:
                center[1] = height / 2
                if center[0] < height / 2:
                    center[0] = height / 2
                elif center[0] > width - height / 2:
                    center[0] = width - height / 2

            size = min(center[0], center[1], input.width - center[0], input.height - center[1], size) * 2

        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2)
        )
        
        # Ensure bbox is within image bounds
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(output.width, bbox[2]),
            min(output.height, bbox[3])
        )
        
        output = output.crop(bbox)
        
        # Add padding to maintain aspect ratio
        width, height = output.size
        if width > height:
            new_height = width
            padding = (width - height) // 2
            padded_output = Image.new('RGBA', (width, new_height), (0, 0, 0, 0))
            padded_output.paste(output, (0, padding))
        else:
            new_width = height
            padding = (height - width) // 2
            padded_output = Image.new('RGBA', (new_width, height), (0, 0, 0, 0))
            padded_output.paste(output, (padding, 0))
        
        # Resize padded image to target size
        # padded_output = padded_output.resize((resolution, resolution), Image.Resampling.LANCZOS)
        padded_output = torch.from_numpy(np.array(padded_output).astype(np.float32)) / 255
        padded_output = F.interpolate(padded_output.unsqueeze(0).permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        
        # Final processing
        output = padded_output.cpu().numpy()
        if no_background:
            output = np.dstack((
                output[:, :, :3] * (output[:, :, 3:4] > 0.8),  # RGB channels premultiplied by alpha
                output[:, :, 3]                         # Original alpha channel
            ))
        output = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
        
        return output

    def _lazy_load_birefnet(self):
        """Lazy loading of the BiRefNet model"""
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoModelForImageSegmentation
        self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True
        ).to(self.device)
        self.birefnet_model.eval()

    def _get_birefnet_mask(self, image: Image.Image) -> np.ndarray:
        """Get object mask using BiRefNet"""
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_images = transform_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.birefnet_model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)

        return (mask_np > 128).astype(np.uint8)

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]], w_layernorm=True) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            image = F.interpolate(image, self.default_image_resolution, mode='bilinear', align_corners=False)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.default_image_resolution, self.default_image_resolution), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        if w_layernorm:
            features = F.layer_norm(features, features.shape[-1:])
        return features
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def encode_slat(
        self,
        slat: sp.SparseTensor,
    ):
        ret = {}
        slat = self.models['slat_encoder'](slat, sample_posterior=False)
        ret['slat'] = slat
        return ret

    @torch.no_grad()
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        ret['slat'] = slat
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def get_input(self, batch_data):
        std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)

        images = batch_data['source_image']
        cond = self.encode_image(images)
        if random.random() > 0.5:
            cond = torch.zeros_like(cond)

        target_feats = batch_data['target_feats']
        target_coords = batch_data['target_coords']
        targets = sp.SparseTensor(target_feats, target_coords).to(self.device)
        targets = (targets - mean) / std

        noise = sp.SparseTensor(
            feats=torch.randn_like(target_feats).to(self.device),
            coords=target_coords.to(self.device),
        )
        return targets, cond, noise
    
    @torch.no_grad()
    def batch_run(self, 
        images: torch.Tensor,
        coords: torch.Tensor,
        feats: torch.Tensor = None,
        ref_images: torch.Tensor = None,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = [],
        ) -> Dict[str, Any]:
        # Get image conditioning
        
        flow_model = self.models['slat_flow_model']
        torch.manual_seed(seed)

        if ref_images is not None:
            cond = {
                'cond': self.encode_image(images),
                'neg_cond': self.encode_image(ref_images),
            }
        else:
            cond = self.get_cond(images)

        # Process input data
        if feats is not None:
            init_slat = sp.SparseTensor(feats, coords).to(self.device)
            std = torch.tensor(self.slat_normalization['std'])[None].to(init_slat.device)
            mean = torch.tensor(self.slat_normalization['mean'])[None].to(init_slat.device)
            init_slat = (init_slat - mean) / std
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
                coords=coords,
            )
            noise = self.slat_sampler._xstart_to_x_t(init_slat, 1, noise)
        else:
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
                coords=coords.to(self.device),
            )

        # Sample structured latent
        slat_sampler_params = {**self.slat_sampler_params, **slat_sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **slat_sampler_params,
            verbose=False
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        return self.decode_slat(slat, formats)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.slat_flow_model(x, t, cond)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        ref_image: Image.Image = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
        preprocess_image: bool = True,
        init_mesh: trimesh.Trimesh = None,
        coords: torch.Tensor = None,
        normalize_init_mesh: bool = False,
        init_resolution: int = 62,
        init_stride: int = 4
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        if ref_image is not None:
            cond = self.encode_image([image, ref_image])
            neg_cond = torch.zeros_like(cond[0:1])
            sparse_cond = slat_cond = {
                'cond': 0.5 * cond[0:1] + 0.5 * cond[1:2],
                'neg_cond': neg_cond,
            }
        else:
            sparse_cond = slat_cond = self.get_cond([image])

        torch.manual_seed(seed)
        if init_mesh is not None:
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(init_mesh.vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(init_mesh.faces)
            if normalize_init_mesh:
                vertices = np.asarray(mesh_o3d.vertices)
                init_mesh = normalize_trimesh(init_mesh)
                center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
                vertices = vertices - center
                diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
                vertices = vertices / diag
                mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            
            vertices = np.clip(np.asarray(mesh_o3d.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh_o3d,
                voxel_size=1/64,
                min_bound=(-0.5, -0.5, -0.5),
                max_bound=(0.5, 0.5, 0.5)
            )
            
            voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            coords = torch.cat([torch.zeros(len(voxel_indices), 1), torch.tensor(voxel_indices)], dim=1).int().to(self.device)
        elif coords is not None:
            coords = coords
        else:
            coords = self.sample_sparse_structure(sparse_cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(slat_cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    def configure_optimizers(self):
        params = list(self.slat_flow_model.parameters())
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return opt

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TrellisVGGTLORATo3DPipeline(TrellisImageTo3DPipeline):

    def get_input(self, batch_data):

        std = torch.tensor(self.slat_normalization['std'])[None].to(self.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(self.device)

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

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.VGGT_dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                aggregated_tokens_list, ps_idx = self.VGGT_model.aggregator(images)
        
        image_cond = self.encode_image(images.reshape(b*n,c,h,w)).reshape(b, n, -1, 1024)[:, :, 5:]
        cond = self.multiview_cond_3d(aggregated_tokens_list, image_cond)
        if random.random() < 0.3:
            cond = torch.zeros_like(cond)

        target_feats = batch_data['target_feats']
        target_coords = batch_data['target_coords']
        targets = sp.SparseTensor(target_feats, target_coords).to(self.device)
        targets = (targets - mean) / std
    
        noise = sp.SparseTensor(
            feats=torch.randn_like(target_feats).to(self.device),
            coords=target_coords.to(self.device),
        )
        return targets, cond, noise

    def get_ss_cond(self, image_cond: torch.Tensor, aggregated_tokens_list: List, num_samples: int) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.multiview_cond(aggregated_tokens_list, image_cond)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def get_cond(self, image_cond: torch.Tensor, aggregated_tokens_list: List, num_samples: int, data_type=torch.float32) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.multiview_cond_3d(aggregated_tokens_list, image_cond)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    @torch.no_grad()
    def vggt_feat(self, image: Union[torch.Tensor, list[Image.Image]]) -> List:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            image = F.interpolate(image, self.default_image_resolution, mode='bilinear', align_corners=False)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.default_image_resolution, self.default_image_resolution), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.VGGT_dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                aggregated_tokens_list, _ = self.VGGT_model.aggregator(image[None])
        
        return aggregated_tokens_list

    def whole_run(
        self,
        image: Union[torch.Tensor, list[Image.Image]],
        coords: torch.Tensor = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> sp.SparseTensor:

        aggregated_tokens_list = self.vggt_feat(image)
        b, n, _, _ = aggregated_tokens_list[0].shape
        image_cond = self.encode_image(image).reshape(b, n, -1, 1024)
        if coords is None:
            ss_flow_model = self.models['sparse_structure_flow_model']
            ss_cond = self.get_ss_cond(image_cond[:, :, 5:], aggregated_tokens_list, num_samples)
            # Sample structured latent
            ss_sampler_params = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}
            reso = ss_flow_model.resolution
            ss_noise = torch.randn(num_samples, ss_flow_model.in_channels, reso, reso, reso).to(self.device)
            ss_slat = self.sparse_structure_sampler.sample(
                ss_flow_model,
                ss_noise,
                **ss_cond,
                **ss_sampler_params,
                verbose=True
            ).samples

            decoder = self.models['sparse_structure_decoder']
            coords = torch.argwhere(decoder(ss_slat)>0)[:, [0, 2, 3, 4]].int()

        import torchvision
        for i, img in enumerate(image):
            if isinstance(img, Image.Image):
                img.save('tmp_{i}.png'.format(i=i))
            else:
                torchvision.utils.save_image(img, 'tmp_{i}.png'.format(i=i))
            
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords[:,1:].cpu().numpy())
        o3d.io.write_point_cloud('tmp.ply', pcd)

        
        cond = {
            'cond': image_cond.reshape(n, -1, 1024),
            'neg_cond': torch.zeros_like(image_cond.reshape(n, -1, 1024))[:1],
        }
        
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(image), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @staticmethod
    def from_pretrained(path: str, whole_pipeline: bool=False, resume_path=None) -> "TrellisVGGTLORATo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisVGGTLORATo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args
        
        new_pipeline.multiview_cond_3d = ModulatedMultiViewCond(
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
        
        if not whole_pipeline:
            for key in ['sparse_structure_flow_model', 'slat_encoder', 'slat_decoder_gs', 'slat_decoder_mesh', 'slat_decoder_rf']:
                if key in new_pipeline.models:
                    del new_pipeline.models[key]
                    
            if hasattr(new_pipeline, 'sparse_structure_flow_model'):
                del new_pipeline.sparse_structure_flow_model

            if resume_path is not None:
                states = torch.load(resume_path)
                if 'state_dict' in states:
                    states = states['state_dict']
                new_pipeline.models['slat_flow_model'].load_state_dict({k.replace(f"slat_flow_model.", ""): v for k, v in states.items()}, False)
                new_pipeline.multiview_cond_3d.load_state_dict({k.replace(f"multiview_cond_3d.", ""): v for k, v in states.items()}, False)

        else:
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
            new_pipeline.models['ss_encoder'] = models.from_pretrained("weights/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
            new_pipeline.models['ss_encoder'].eval()

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
        
        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])
        
        # new_pipeline.max_num_points = 6000

        return new_pipeline
    
    def configure_optimizers(self):
        trellis_lora_params = [p for p in self.slat_flow_model.parameters() if p.requires_grad]
        params = list(self.multiview_cond_3d.parameters()) + trellis_lora_params
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        return opt
