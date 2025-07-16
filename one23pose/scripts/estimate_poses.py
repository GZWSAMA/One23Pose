from fpose.estimater import *

def estimate_poses(npz_path, query_image_names, query_depth_names, query_mask_names, query_intrinsics, scaled_model_path, output_dir, debug=0, est_refine_iter=5):
    #estimate the poses of the query images
    debug_dir = output_dir
    mesh = trimesh.load(scaled_model_path, force='mesh')
    
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    poses = []
    rgbs = []
    depths = []
    for frame_id,query_image_name in enumerate(query_image_names):
        color = cv2.imread(query_image_name)
        # depth = reader.get_depth(i)
        depth = cv2.imread(query_depth_names[frame_id],-1)/1e3
        # mask = reader.get_mask(0).astype(bool)
        mask = cv2.imread(query_mask_names[frame_id],-1)
        K = np.array(query_intrinsics[frame_id])
        if len(mask.shape)==3:
            for c in range(3):
                if mask[...,c].sum()>0:
                    mask = mask[...,c]
                    break
        mask = mask.astype(bool)
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
        poses.append(pose.reshape(4, 4))

        center_pose = pose@np.linalg.inv(to_origin)
        rgb_vis, dep_vis = draw_posed_3d_box_with_depth(K, img=color, depth=depth, ob_in_cam=center_pose, bbox=bbox)
        rgb_new, dep_new = draw_xyz_axis_with_depth(rgb_vis, depth=dep_vis, ob_in_cam=center_pose, scale=0.3, K=K, thickness=3, transparency=0, is_input_rgb=True)
        rgb_new = np.transpose(rgb_new, (2, 0, 1))/255
        rgbs.append(rgb_new)
        depths.append(dep_new)

    data = dict(np.load(npz_path))
    # 替换指定字段
    data['depths'] = depths
    data['video'] = rgbs
    np.savez(npz_path, **data)
    return poses  # Return the list of estimated poses for each image