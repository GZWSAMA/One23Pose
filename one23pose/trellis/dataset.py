import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import random
import pdb
import PIL.Image as Image
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
from trellis.modules import sparse as sp
from torch.utils.data import DataLoader
from trellis.utils import render_utils
import utils3d

def get_camera_param_list(num_frames=40, r=2, fov=40, inverse_direction=False, pitch=-1):
    if inverse_direction:
        yaws = torch.linspace(3.1415, -3.1415, num_frames)
        # pitch = 0.25 + 0.5 * torch.sin(torch.linspace(2 * 3.1415, 0, num_frames))
    else:
        yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    if pitch != -1:
        pitch = pitch * torch.ones(num_frames)
    else:
        pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov, 'cpu')
    return extrinsics, intrinsics

class NpzDatasetAug(Dataset):
    def __init__(self, json_files=None, stage=2,
                 source_aug=None, source_aug_prob=0.1, 
                 task='normal', random_sample=-1, num_views=1):
        self.data = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                data_dir = os.path.dirname(json_file)
                for line in f:
                    data_term = json.loads(line)
                    data_term['data_dir'] = data_dir
                    self.data += [data_term]
        self.to_Tensor = T.ToTensor()
        self.source_aug = source_aug
        self.source_aug_prob = source_aug_prob
        self.task = task
        self.random_sample = random_sample
        self.num_views = num_views
        if self.num_views > 1:
            self.extrinsics, self.intrinsics = get_camera_param_list()
        self.stage = stage

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception as e:
                # print("Error loading item", e)
                idx = random.randint(0, len(self.data) - 1)
                
    def _load_item(self, idx):
        item = self.data[idx]
        data_dir = item['data_dir']
        if self.stage == 1:
            ss_data_dir = os.path.join(os.path.dirname(item['data_dir']), 'data_dreamverse2_sslatent')
            
            _target_data = np.load(os.path.join(ss_data_dir, item['target_slat'].replace('slat.npz', 'ss.npz')))
            target_ss_latents = torch.from_numpy(_target_data['mean'])
        
        else:   
            _target_data = np.load(os.path.join(data_dir, item['target_slat']))
            target_feats, target_coords = torch.from_numpy(_target_data['feats']), torch.from_numpy(_target_data['coords'])
            if self.random_sample > 0:
                random_sample = min(self.random_sample, target_feats.shape[0])
                idxs = np.random.choice(target_feats.shape[0], random_sample, replace=False)
                target_feats = target_feats[idxs]
                target_coords = target_coords[idxs]

        # Load and convert images to tensors
        source_data = np.load(os.path.join(data_dir, item['source_image']))
        _data_len = len(source_data[self.task])

        random_idx = random.randint(0, _data_len - 1)
        if self.num_views > 1:
            if random.random() < 1:
                rand_interval = random.randint(0,4)
                random_idx = [(random_idx + i * rand_interval) % _data_len for i in range(self.num_views)]
            else:
                random_idx = np.random.choice(_data_len, self.num_views, replace=False).tolist()
            # random_idx = np.random.choice(_data_len, self.num_views, replace=False).tolist()
            source = torch.stack([self.to_Tensor(source_data[self.task][i]).float() for i in random_idx])
            ref = torch.stack([self.to_Tensor(source_data['colors'][i]).float() for i in random_idx])
            # ref = ref.permute(0,2,3,1)
            # ref[(ref[...,0] == 0.) & (ref[...,1] == 0.) & (ref[...,2] == 0.)] = 1.
            # ref = ref.permute(0,3,1,2)
            sample_extrinsics = torch.stack([self.extrinsics[i] for i in random_idx])
            sample_intrinsics = torch.stack([self.intrinsics[i] for i in random_idx])
        else:
            source = self.to_Tensor(source_data[self.task][random_idx]).float()
            ref = self.to_Tensor(source_data['colors'][random_idx]).float()
            # ref = ref.permute(1,2,0)
            # ref[(ref[...,0] == 0.) & (ref[...,1] == 0.) & (ref[...,2] == 0.)] = 1.
            # ref = ref.permute(2,0,1)

        if source.max() > 1:
            source /= 255.0
        if ref.max() > 1:
            ref /= 255.0

        # Apply source augmentation if specified
        if self.source_aug is not None and random.random() < self.source_aug_prob:
            source = self.source_aug(source)

        # Return with random condition
        data_dict = dict(source_image=source, ref_image=ref)
        if self.stage == 1:
            data_dict.update(target_ss_latents=target_ss_latents)
            if self.num_views > 1:
                # data_dict.update(batch_extrinsics=sample_extrinsics, batch_intrinsics=sample_intrinsics, volume_resolution=volume_resolution)
                data_dict.update(batch_extrinsics=sample_extrinsics, batch_intrinsics=sample_intrinsics)
        else:
            data_dict.update(target_feats=target_feats, target_coords=target_coords)
            if self.num_views > 1:
                # data_dict.update(batch_extrinsics=sample_extrinsics, batch_intrinsics=sample_intrinsics, volume_resolution=volume_resolution)
                data_dict.update(batch_extrinsics=sample_extrinsics, batch_intrinsics=sample_intrinsics)
        return data_dict

def blender_world_normal_2_camera(normals_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Transform normal from world space to camera space.
    :param normal: The normal in world space.
    :param c2w: The camera to world matrix.
    :return: The normal in camera space.
    """
    assert len(normals_world.shape) == 3, "Normal must be a 3D vector."
    H, W, C = normals_world.shape

    normals_camera = np.zeros((H, W, C), dtype=np.float32)

    if C == 4:
        normals_world = normals_world[..., :3]

    R_c2w = c2w[:3, :3]
    R_convert = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)

    R_opencv = R_c2w @ R_convert
    R_opencv = R_opencv.T

    transformed_normals = normals_world.reshape(-1, 3).T  
    transformed_normals = R_opencv @ transformed_normals
    transformed_normals = transformed_normals.T
    transformed_normals = transformed_normals.reshape(H, W, 3)

    normals_camera[..., :1] = transformed_normals[..., :1] * 0.5 + 0.5
    normals_camera[..., 2:3] = transformed_normals[..., 1:2] * -0.5 + 0.5
    normals_camera[..., 1:2] = transformed_normals[..., 2:3] * 0.5 + 0.5

    return normals_camera

def blender_depth_2_nocs(depth_map: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    批量将深度图转换为点云。
    
    参数：
    - depth_map: 深度图，形状为 (b, h, w)，每个样本的深度图。
    - K: 相机内参矩阵，形状为 (b, 3, 3)，每个样本的相机内参。
    - pose: 相机外参矩阵，形状为 (b, 4, 4)，每个样本的相机外参。
    
    返回：
    - points: 每个样本对应的点云，形状为 (b, N, 3), N为每个深度图的点数。
    """
    b, h, w = depth_map.shape
    # depth_map[depth_map == 65504.] = 0. 

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # u = u.astype(float)
    # v = v.astype(float)

    # 将深度图转换为相机坐标系下的三维坐标
    Z = depth_map.reshape(b, h * w)  # (b, h * w,)
    X = (u.flatten()[None] - K[:, 0, 2][:,None]) * Z / K[:, 0, 0][:,None]
    Y = (v.flatten()[None] - K[:, 1, 2][:,None]) * Z / K[:, 1, 1][:,None]

    # 组合相机坐标系下的三维坐标
    camera_points = np.stack((X, Y, Z, np.ones_like(Z)), axis=2).reshape(b, h * w, 4, 1)  # (b, h * w, 4, 1)

    # 将相机坐标系下的点转换到世界坐标系
    c2w[:, :3, 1:3] *= -1
    # pose_inv = np.linalg.inv(c2w)[:, None]  # (b, 1, 4, 4)
    pose_inv = c2w[:, None]  # (b, 1, 4, 4)
    world_points = np.matmul(pose_inv, camera_points)  # (b, h * w, 4, 1)
    world_points = world_points[:, :, :3, 0]  # (b, h * w, 3)

    # 将点云添加到列表
    point_clouds = world_points.reshape(b, h, w, 3)
    masks_0 = ((depth_map != 65504.) & (depth_map != 0.) & (depth_map != 1000.5496)).astype(np.uint8)
    masks = []
    for i in range(b):        
        kernel = np.ones((3,3),np.uint8)
        eroded = cv2.erode(
            masks_0[i],  # 确保输入类型正确
            kernel,
            iterations=1,
        )
        masks.append(np.array(eroded))
    masks = np.stack(masks, axis=0)
    point_clouds[masks < 0.5] = 127/255 - 0.5

    return (point_clouds + 0.5).clip(0,1)

import json, pyexr
class JsonDatasetAug(Dataset):
    def __init__(self, json_files=None, stage=2,
                    source_aug=None, source_aug_prob=0.1, 
                    task='normal', random_sample=-1, num_views=1, rayembedding=False):
        self.data = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                data_dir = os.path.dirname(json_file)
                for line in f:
                    data_term = json.loads(line)
                    data_term['data_dir'] = data_dir
                    self.data += [data_term]
        self.to_Tensor = T.ToTensor()
        self.source_aug = source_aug
        self.source_aug_prob = source_aug_prob
        self.task = task
        self.random_sample = random_sample
        self.num_views = num_views
        self.rayembedding = rayembedding
        self.default_image_size = 518
        if self.num_views > 1:
            self.extrinsics, self.intrinsics = get_camera_param_list()
        self.stage = stage

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception as e:
                print("Error loading item", e)
                idx = random.randint(0, len(self.data) - 1)

    def _load_item(self, idx):
        item = self.data[idx]
        data_dir = item['data_dir']
        if self.stage == 1:
            _target_data = np.load(os.path.join(data_dir, item['target_slat'].replace('slat', 'ss')))
            target_ss_latents = torch.from_numpy(_target_data['mean'])
            if torch.isnan(target_ss_latents).any():
                raise ValueError("NaN values found in target_ss_latents")
        else:   
            _target_data = np.load(os.path.join(data_dir, item['target_slat']))
            target_feats, target_coords = torch.from_numpy(_target_data['feats']), torch.from_numpy(_target_data['coords'])
            if self.random_sample > 0:
                random_sample = min(self.random_sample, target_feats.shape[0])
                idxs = np.random.choice(target_feats.shape[0], random_sample, replace=False)
                target_feats = target_feats[idxs]
                target_coords = target_coords[idxs]

            if torch.isnan(target_feats).any() or torch.isnan(target_coords).any():
                raise ValueError("NaN values found in target_feats or target_coords")

        # Load and convert images to tensors
        source_data_path = os.path.join(data_dir, item['source_image'])
        source_data = json.load(open(source_data_path, "r"))
            
        data_dir = os.path.dirname(source_data_path)
        _data_len = len(source_data["frames"])
        random_idx = random.randint(0, _data_len - 1)
        
        if self.num_views > 1:
            # if random.random() < 1.:
            #     rand_interval = random.randint(1,4)
            #     random_idxs = [(random_idx + i * rand_interval) % _data_len for i in range(self.num_views)]
            # else:
            #     random_idxs = np.random.choice(_data_len, self.num_views, replace=False).tolist()
            random_idxs = np.random.choice(_data_len, self.num_views, replace=False).tolist()
            _frame_ids = [source_data["frames"][i]["file_path"].split("/")[-1].split(".")[0] for i in random_idxs]
            sample_extrinsics = np.concatenate([np.array(source_data["frames"][i]["transform_matrix"])[None] for i in random_idxs], axis=0)
            ref_paths = [os.path.join(data_dir, _frame_id + ".png") for _frame_id in _frame_ids]
            ref = torch.cat([TF.to_tensor(Image.open(ref_path)).float()[None] for ref_path in ref_paths], dim=0)
            ref = ref[:,:3] * ref[:,3:] # + (1 - ref[:,3:])
            if self.task == "normal":
                exit("Not implemented")
            elif self.task == "color":
                source=ref
            elif self.task == "nocs":
                source_paths = [os.path.join(data_dir, "depth", _frame_id + "_depth.exr") for _frame_id in _frame_ids]
                source = np.concatenate([pyexr.read(source_path)[...,0][None] for source_path in source_paths], axis=0)
                cam_fov = torch.cat([torch.tensor(source_data["frames"][random_idx]["camera_angle_x"])[None] for random_idx in random_idxs], dim=0)
                sample_intrinsics = utils3d.torch.intrinsics_from_fov_xy(cam_fov, cam_fov)
                sample_intrinsics[..., :2, :] *= source.shape[1]
                source = blender_depth_2_nocs(source, sample_intrinsics.numpy(), sample_extrinsics)
                sample_extrinsics = torch.inverse(torch.from_numpy(sample_extrinsics).float())
                sample_intrinsics[..., :2, :] /= source.shape[1]
                source = torch.from_numpy(source).permute(0, 3, 1, 2).float()
        else:
            _light_name = list(source_data["frames"].keys())[random_idx]
            _light_data = source_data["frames"][_light_name]
            random_view_idx = random.randint(0, len(_light_data) - 1)
            _frame_data = _light_data[random_view_idx]
            _frame_id = f"{random_view_idx:03d}_hdri_{random_idx:02d}"
            _gt_frame_id = f"{random_view_idx:03d}"
            _transform_matrix = _frame_data["transform_matrix"]
            _transform_matrix = np.array(_transform_matrix)
            ref_path = os.path.join(data_dir, _frame_id + ".png")
            ref = Image.open(ref_path)
            ref = TF.to_tensor(ref).float()
            ref = ref[:3] * ref[3] # + (1 - ref[3])
            if self.task == "normal":
                source_path = os.path.join(data_dir, "normal", _gt_frame_id + "_normal.exr")
                source = pyexr.read(source_path)
                source = blender_world_normal_2_camera(source, _transform_matrix)
                source = torch.from_numpy(source).permute(2, 0, 1).float()
            elif self.task == "color":
                source=ref
            elif self.task == "nocs":
                source_path = os.path.join(data_dir, "depth", _gt_frame_id + "_depth.exr")
                source = pyexr.read(source_path)[...,0]
                cam_fov = _frame_data["camera_angle_x"]
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(cam_fov), torch.tensor(cam_fov))
                intrinsics[..., :2, :] *= source.shape[0]
                source = blender_depth_2_nocs(source[None], intrinsics.numpy()[None], _transform_matrix[None])[0]
                source = torch.from_numpy(source).permute(2, 0, 1).float()
            if self.rayembedding:
                blender_cam_fov = torch.tensor(_frame_data["camera_angle_x"]).float()
                blender_cam_transform = torch.from_numpy(_transform_matrix).float()

        # Apply source augmentation if specified
        if self.source_aug is not None and random.random() < self.source_aug_prob:
            source = self.source_aug(source)

        # Return with random condition
        data_dict = dict(source_image=source, ref_image=ref)
        if self.stage == 1:
            data_dict.update(target_ss_latents=target_ss_latents)
            if self.num_views > 1:
                data_dict.update(batch_extrinsics=sample_extrinsics, batch_intrinsics=sample_intrinsics)
        else:
            data_dict.update(target_feats=target_feats, target_coords=target_coords)
            if self.num_views > 1:
                data_dict.update(batch_extrinsics=sample_extrinsics, batch_intrinsics=sample_intrinsics)
        if self.rayembedding:
            data_dict.update(blender_cam_fov=blender_cam_fov, blender_cam_transform=blender_cam_transform)
        return data_dict

def custom_collate(batch):
    """
    Custom collate function that handles sparse tensors along with other batch elements.
    
    Args:
        batch: List of dictionaries containing 'target_feats', 'target_coords', 'condition_feats', 'condition_coords', 'source' and 'ref'
        
    Returns:
        Dictionary with batched data
    """
    # Initialize lists to hold the batched data
    if 'target_ss_latents' in batch[0]:
        batched_ss_latents = []
    else:
        batched_feats = []
        batched_coords = []
    batched_source = []
    batched_ref = []
    if 'batch_extrinsics' in batch[0] and 'batch_intrinsics' in batch[0]:
        batched_extrinsics = []
        batched_intrinsics = []
    if 'blender_cam_fov' in batch[0] and 'blender_cam_transform' in batch[0]:
        batched_blender_cam_fov = []
        batched_blender_cam_transform = []
    batch_size = len(batch)
    for idx, sample in enumerate(batch):
        # Handle target sparse tensor components
        if 'target_ss_latents' in sample:
            ss_latents = sample['target_ss_latents']
            batched_ss_latents.append(ss_latents)
        else:       
            feats = sample['target_feats']
            coords = sample['target_coords'][..., 1:]
            
            # Add batch dimension to coordinates
            batch_coords = torch.full((coords.shape[0], 1), idx, dtype=coords.dtype)
            
            batched_feats.append(feats)
            batched_coords.append(torch.cat([batch_coords, coords], dim=1))
        
        # Handle other batch elements
        batched_source.append(sample['source_image'])
        batched_ref.append(sample['ref_image'])

        if 'batch_extrinsics' in sample and 'batch_intrinsics' in sample:
            batched_extrinsics.append(sample['batch_extrinsics'])
            batched_intrinsics.append(sample['batch_intrinsics'])
        
        if 'blender_cam_fov' in sample and 'blender_cam_transform' in sample:
            batched_blender_cam_fov.append(sample['blender_cam_fov'])
            batched_blender_cam_transform.append(sample['blender_cam_transform'])
    
    # Stack source tensors
    batched_source = torch.stack(batched_source, dim=0)
    batched_ref = torch.stack(batched_ref, dim=0)
    batch_data_dict = {'source_image': batched_source,
                       'ref_image': batched_ref}
    
    # Concatenate target features and coordinates
    if 'target_ss_latents' in batch[0]:
        batched_ss_latents = torch.stack(batched_ss_latents, dim=0)
        batch_data_dict['target_ss_latents'] = batched_ss_latents
    else:
        batched_feats = torch.cat(batched_feats, dim=0)
        batched_coords = torch.cat(batched_coords, dim=0)
        batch_data_dict['target_feats'] = batched_feats
        batch_data_dict['target_coords'] = batched_coords

    if 'batch_extrinsics' in batch[0] and 'batch_intrinsics' in batch[0]:
        batched_extrinsics = torch.stack(batched_extrinsics, dim=0)
        batched_intrinsics = torch.stack(batched_intrinsics, dim=0)
        batch_data_dict['batch_extrinsics'] = batched_extrinsics
        batch_data_dict['batch_intrinsics'] = batched_intrinsics
    
    if 'blender_cam_fov' in batch[0] and 'blender_cam_transform' in batch[0]:
        batched_blender_cam_fov = torch.stack(batched_blender_cam_fov, dim=0)
        batched_blender_cam_transform = torch.stack(batched_blender_cam_transform, dim=0)
        batch_data_dict['blender_cam_fov'] = batched_blender_cam_fov
        batch_data_dict['blender_cam_transform'] = batched_blender_cam_transform

    return batch_data_dict

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # from augmentation import RandomOcclusion
    # Create output directory
    output_dir = './dataset_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    # dataset = NpzDatasetAug(
    #     json_files=['./data/train.jsonl'],
    #     random_sample=30720,
    #     num_views=4,
    #     task='nocs',
    # )
    # dataset = JsonDatasetAug(json_files=['./render_data/train_randombg.jsonl'], task='color', num_views=1, source_aug=RandomOcclusion(occlusion_prob=1.0))
    dataset = JsonDatasetAug(json_files=['./objaverse_data/train_pbr_eevee_evenbg.jsonl'], num_views=6,task='nocs')
    
    sample = dataset[2]
    print(f"Dataset size: {len(dataset)}")
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
        f.write(f"Dataset size: {len(dataset)}\n")
        
        sample = dataset[0]
        f.write("\nSample keys: " + str(sample.keys()) + "\n")
        f.write("\nShapes:\n")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                f.write(f"{key}: {value.shape}\n")
        
    # Save sample visualizations
    for i in range(min(5, len(dataset))):  # Save first 5 samples
        fig = plt.figure(figsize=(15, 5))
        sample = dataset[i]
        
        plt.subplot(132)
        plt.imshow(sample['source_image'].permute(1, 2, 0).numpy())
        plt.title('Source')
        plt.axis('off')
        
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(133, projection='3d')
        coords = sample['target_coords'][..., 1:].numpy() # [N, 3]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)
        ax.set_xlim([0, 63])
        ax.set_ylim([0, 63])
        ax.set_zlim([0, 63])
        
        plt.savefig(os.path.join(output_dir, f'sample_{i}.png'), 
                   bbox_inches='tight', 
                   pad_inches=0.1,
                   dpi=300)
        plt.close(fig)
    
    print(f"Results saved to {output_dir}")