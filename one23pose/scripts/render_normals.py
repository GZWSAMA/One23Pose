import cv2
import os
import numpy as np
import torch
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras,
                                PointLights, RasterizationSettings, BlendParams,
                                MeshRenderer, MeshRasterizer, SoftPhongShader)
from pytorch3d.io import load_objs_as_meshes


import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    PointLights
)
from pytorch3d.structures import Meshes

import numpy as np

def compute_fov(K, width, height, degrees=False):
    """
    根据相机内参矩阵 K 和图像尺寸计算 FoV
    
    参数:
    K (np.ndarray): 3x3 相机内参矩阵
    width (int): 图像宽度（像素）
    height (int): 图像高度（像素）
    degrees (bool): 是否返回角度，默认 True
    
    返回:
    fovx (float): 水平视场角
    fovy (float): 垂直视场角
    """
    fx = K[0, 0]
    fy = K[1, 1]

    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))

    if degrees:
        fovx = np.degrees(fovx)
        fovy = np.degrees(fovy)

    return fovx, fovy

def render_normal_map(mesh_path, model_pose, width=640, height=480, fov=1, device='cpu'):
    """
    渲染指定位姿下的模型法线图
    :param mesh_path: str, 模型路径 (.obj)
    :param model_pose: numpy array [4,4], 世界坐标到相机坐标的变换矩阵 (OpenCV 格式)
    :param width: 输出图像宽度
    :param height: 输出图像高度
    :param fov: 视场角（弧度）
    :param device: 'cpu' or 'cuda'
    :return: normal_map: numpy array of shape (H, W, 3) in [0, 255]
    """

    # ---------------------------
    # Step 1: Load Mesh
    # ---------------------------
    mesh = load_objs_as_meshes([mesh_path], device=device)

    # ---------------------------
    # Step 2: Convert OpenCV pose to PyTorch3D camera matrix
    # ---------------------------
    rotation_matrix = model_pose[:3, :3]
    tvec = model_pose[:3, 3]

    world_2_cam_render = np.eye(4, dtype=np.float32)
    world_2_cam_render[:3, :3] = np.linalg.inv(rotation_matrix)
    world_2_cam_render[3, :3] = tvec
    world_2_cam_render[:, :2] = -world_2_cam_render[:, :2]  # PyTorch3D 坐标系适配

    camera_pose = torch.tensor(world_2_cam_render, device=device).unsqueeze(0)  # (1, 4, 4)

    R = camera_pose[:, :3, :3]
    T = camera_pose[:, 3, :3]

    # ---------------------------
    # Step 3: Setup Camera
    # ---------------------------
    focal_length = torch.ones(1, 1, device=device) * 0.5 * width / np.tan(fov / 2)
    principal_point = torch.tensor([[width / 2, height / 2]], device=device).repeat(1, 1)
    image_size = torch.tensor([[height, width]], device=device).repeat(1, 1)

    cameras = PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        in_ndc=False,
        device=device
    )

    # ---------------------------
    # Step 4: Set up Rasterizer
    # ---------------------------
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    # ---------------------------
    # Step 5: Rasterize and Interpolate Normals
    # ---------------------------
    fragments = rasterizer(mesh)
    pix_to_face = fragments.pix_to_face  # (N, H, W, K)
    bary_coords = fragments.bary_coords    # (N, H, W, K, 1)
    mesh_faces_normals = mesh.faces_normals_packed()  # (F, 3)

    N, H, W, K = pix_to_face.shape

    # Flatten for indexing
    pix_to_face_flat = pix_to_face.view(N, H * W, K)
    bary_coords_flat = bary_coords.view(N, H * W, K, 3)

    # Get valid normals using face indices
    mask = pix_to_face_flat >= 0
    normals = torch.zeros((N, H * W, K, 3), device=device)
    for i in range(K):
        valid_idx = pix_to_face_flat[:, :, i][mask[:, :, i]]
        if len(valid_idx) > 0:
            normals[:, :, i, :][mask[:, :, i]] = mesh_faces_normals[valid_idx]

    # Interpolate per-pixel normals
    pixel_normals = (normals * bary_coords_flat).sum(dim=2)  # (N, H*W, 3)
    pixel_normals = pixel_normals.view(N, H, W, 3)  # (N, H, W, 3)

    # Normalize and convert to RGB
    pixel_normals = torch.nn.functional.normalize(pixel_normals, dim=-1)
    pixel_normals = (pixel_normals + 1.0) / 2.0  # [-1,1] -> [0,1]
    normal_map = (pixel_normals[0].cpu().numpy() * 255).astype(np.uint8)

    return normal_map

def render_normals_to_video(poses, query_image_names, query_intrinsics, scaled_model_path, output_video_path, fps = 15, device='cpu'):
    video_frames = []
    
    for i, pose in enumerate(poses):
        rgb_path = query_image_names[i]
        intrinsic = np.array(query_intrinsics[i])
        pose = np.array(pose)
        
        # 读取 RGB 图像
        rgb_img = cv2.imread(rgb_path)
        height, width = rgb_img.shape[:2]
        fovx, fovy = compute_fov(intrinsic, width, height)
        # 渲染模型法线图
        normal_map = render_normal_map(scaled_model_path, pose, width=width, height=height, fov=fovx, device=device)
        
        # 将法线图叠加到RGB图像上（可选）
        # 注意：normal_map 应该是 [H, W, 3] 的 numpy 数组
        alpha = 0.8  # 调整透明度
        overlay = cv2.addWeighted(normal_map, alpha, rgb_img, 1 - alpha, 0)
        
        # 或者直接使用法线图作为输出帧
        frame = overlay if normal_map is not None else rgb_img
        
        # 添加到视频帧列表中
        video_frames.append(frame)

    # 获取第一帧的尺寸信息
    height, width, layers = video_frames[0].shape
    
    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 写入视频帧
    for frame in video_frames:
        video_writer.write(frame)
    
    # 释放视频写入器
    video_writer.release()

