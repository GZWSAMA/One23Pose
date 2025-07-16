import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.append(os.getcwd())
import trimesh
import copy
import numpy as np
import imageio
import cv2
import logging
from one23pose.locate.fit_object_scale import get_scale
from fpose.estimater import *
from fpose.datareader import *

def get_all_pose(test_scene_dir, mesh, topic, debug, track_refine_iter=8, est_refine_iter=5):
  set_logging_format()
  set_seed(0)
  code_dir = os.path.dirname(os.path.realpath(__file__))

  debug_dir = os.path.join(code_dir, "results", topic)
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/{topic}')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)

    os.makedirs(f'{debug_dir}/{topic}', exist_ok=True)
    np.savetxt(f'{debug_dir}/{topic}/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      # cv2.imshow('1', vis[...,::-1])
      # cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

def get_single_pose(raw_img, depth, mask, intrinsic, mesh, topic, num, debug, est_refine_iter=5):
    set_logging_format()
    set_seed(0)
    code_dir = os.path.dirname(os.path.realpath(__file__))

    debug_dir = os.path.join(code_dir, "results", topic)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

#   reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

    # color = reader.get_color(i)
    color = cv2.imread(raw_img)
    # depth = reader.get_depth(i)
    depth = cv2.imread(depth,-1)/1e3
    # mask = reader.get_mask(0).astype(bool)
    mask = cv2.imread(mask,-1)
    K = intrinsic
    if len(mask.shape)==3:
        for c in range(3):
            if mask[...,c].sum()>0:
                mask = mask[...,c]
                break
    mask = mask.astype(bool)
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
    
    os.makedirs(f'{debug_dir}/{topic}', exist_ok=True)
    np.savetxt(f'{debug_dir}/{topic}/{num}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
      # cv2.imshow('1', vis[...,::-1])
      # cv2.waitKey(1)

    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{num}.png', vis)

    return pose.reshape(4,4)

def recover_scale(mesh_file, depth_file, raw_img, mask_file, intrinsic_file, topic, out_dir):
    mesh = trimesh.load(mesh_file, force='mesh')
    mesh_rotated = mesh.copy()

    scales = []
    poses = []

    #进行模型缩放比例和位姿的确定
    M, intrinsic_numpy, scale = get_scale(mesh_file, depth_file, raw_img, mask_file, out_dir, intrinsic_file)
    scales.append(scale)
    scale_matrix = np.array([[scale, 0, 0, 0],
                             [0, scale, 0, 0],
                             [0, 0, scale, 0],
                             [0, 0, 0, 1]])
    logging.info(f"initial scale: {scale}")

    #进行模型放缩
    mesh.apply_transform(scale_matrix)
    mesh_rotated.apply_transform(scale_matrix)

    #按照初次缩放的位姿进行refine
    for i in range(3):
        #使用Foundationpose进行位姿估计
        pose = get_single_pose(raw_img, depth_file, mask_file, intrinsic_numpy, mesh_rotated, topic, i, debug=0, est_refine_iter=3)
        poses.append(pose)

        #进行模型旋转
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = pose[0:3, 0:3]
        mesh_rotated.apply_transform(rotation_matrix)
        mesh_rotated.export(os.path.join(out_dir, "rotated_model.obj"))
        rotated_mesh_dir = os.path.join(out_dir, "rotated_model.obj")

        #根据得到的位姿再次进行转换矩阵估计
        scale_output = os.path.join(out_dir, f"iteration_{i+1}")
        os.makedirs(scale_output, exist_ok=True)
        M, intrinsic_numpy, scale = get_scale(rotated_mesh_dir, depth_file, raw_img, mask_file, scale_output, intrinsic_file)
        if scale > 1.5 or scale < 0.5:
           i = i - 1
           continue
        scales.append(scale)
        scale_matrix = np.array([[scale, 0, 0, 0],
                                [0, scale, 0, 0],
                                [0, 0, scale, 0],
                                [0, 0, 0, 1]])
        logging.info(f"iter{i} scale: {scale}")

        #进行模型缩放
        mesh.apply_transform(scale_matrix)
        mesh_rotated.apply_transform(scale_matrix)

    print(f"final scale: {scales}")
    print(f"final pose: {poses}")
    final_scale = 1.0
    for scale in scales:
      final_scale *= scale
    return mesh, pose, final_scale

if __name__ == "__main__":
    #输入mesh，ref_img, mask, depth, intrinsic_file
    mesh_file = "/baai-cwm-1/baai_cwm_ml/cwm/yuhao.duan/gz/TRELLIS/output/custom_datasets/gun/original_model/output.obj"
    depth_file = "/baai-cwm-1/baai_cwm_ml/cwm/yuhao.duan/gz/blenderproc_geng/output/custom_datasets/scene_easy_small/gun/bop_with_true_imgs/lm/train_pbr/000000/depth/000308.png"
    raw_img = "/baai-cwm-1/baai_cwm_ml/cwm/yuhao.duan/gz/blenderproc_geng/output/custom_datasets/scene_easy_small/gun/bop_with_true_imgs/lm/train_pbr/000000/rgb/000308.png"
    mask_file = "/baai-cwm-1/baai_cwm_ml/cwm/yuhao.duan/gz/blenderproc_geng/output/custom_datasets/scene_easy_small/gun/bop_with_true_imgs/lm/train_pbr/000000/mask/000308_000000.png"
    out_path = "results"
    intrinsic_file = "/baai-cwm-1/baai_cwm_ml/cwm/yuhao.duan/gz/blenderproc_geng/output/custom_datasets/scene_easy_small/gun/bop_with_true_imgs/lm/train_pbr/000000/scene_camera.json"
    topic = "test"
    out_dir = os.path.join(out_path, topic)
    ref_img = os.path.join(out_dir, "ref_img.png")

    os.system(f'rm -rf {out_dir}/* && mkdir -p {out_dir}/track_vis {out_dir}/{topic}')
    os.makedirs(out_dir, exist_ok=True)

    recover_scale(mesh_file, depth_file, raw_img, mask_file, intrinsic_file, topic, out_dir, ref_img)

        

    #进行最终的位姿估计
