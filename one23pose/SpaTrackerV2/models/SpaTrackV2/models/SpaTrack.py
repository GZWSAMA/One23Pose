#python 
"""
SpaTrackerV2, which is an unified model to estimate 'intrinsic',
'video depth', 'extrinsic' and '3D Tracking' from casual video frames.

Contact: DM yuxixiao@zju.edu.cn
"""

import os
import numpy as np
from typing import Literal, Union, List, Tuple, Dict
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from depth anything v2
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from einops import rearrange
from models.monoD.depth_anything_v2.dpt import DepthAnythingV2
from models.moge.model.v1 import MoGeModel
import copy 
from functools import partial
from models.SpaTrackV2.models.tracker3D.TrackRefiner import TrackRefiner3D
import kornia
from models.SpaTrackV2.utils.model_utils import sample_features5d
import utils3d
from models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import depth_to_points_colmap, get_nth_visible_time_index
from models.SpaTrackV2.models.utils import pose_enc2mat, matrix_to_quaternion, get_track_points, normalize_rgb
import random

class SpaTrack2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        loggers: list,   # include [ viz, logger_tf, logger] 
        backbone_cfg,
        Track_cfg=None,
        chunk_size=24,
        ckpt_fwd: bool = False,
        ft_cfg=None,
        resolution=518,
        max_len=600,  # the maximum video length we can preprocess,
        track_num=768,
        moge_as_base=False,
    ):
    
        self.chunk_size = chunk_size
        self.max_len = max_len
        self.resolution = resolution
        # config the T-Lora Dinov2
        #NOTE: initial the base model
        base_cfg = copy.deepcopy(backbone_cfg)
        backbone_ckpt_dir = base_cfg.pop('ckpt_dir', None)

        super(SpaTrack2, self).__init__()
        if moge_as_base:
            if os.path.exists(backbone_ckpt_dir)==False:
                base_model = MoGeModel.from_pretrained('Ruicheng/moge-vitl')
            else:
                checkpoint = torch.load(backbone_ckpt_dir, map_location='cpu', weights_only=True)
                base_model = MoGeModel(**checkpoint["model_config"])
                base_model.load_state_dict(checkpoint['model'])
        else:
            base_model = None
        # avoid the base_model is a member of SpaTrack2
        object.__setattr__(self, 'base_model', base_model)

        # Tracker model
        self.Track3D = TrackRefiner3D(Track_cfg)
        track_base_ckpt_dir = Track_cfg["base_ckpt"]
        if os.path.exists(track_base_ckpt_dir):
            track_pretrain = torch.load(track_base_ckpt_dir)
            self.Track3D.load_state_dict(track_pretrain, strict=False)
        
        # wrap the function of make lora trainable
        self.make_paras_trainable = partial(self.make_paras_trainable,
                                            mode=ft_cfg["mode"],
                                            paras_name=ft_cfg["paras_name"])
        self.track_num = track_num

    def make_paras_trainable(self, mode: str = 'fix', paras_name: List[str] = []):
        # gradient required for the lora_experts and gate
        for name, param in self.named_parameters():
            if any(x in name for x in paras_name):
                if mode == 'fix':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                if mode == 'fix':
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params/total_params*100:.2f}%")
        
    def ProcVid(self, 
                        x: torch.Tensor):
        """
        split the video into several overlapped windows.
        
        args:
            x: the input video frames.   [B, T, C, H, W]
        outputs:
            patch_size: the patch size of the video features
        raises:
            ValueError: if the input video is longer than `max_len`.
        
        """
        # normalize the input images
        num_types = x.dtype
        x = normalize_rgb(x, input_size=self.resolution)
        x = x.to(num_types)
        # get the video features
        B, T, C, H, W = x.size()
        if T > self.max_len:
            raise ValueError(f"the video length should no more than {self.max_len}.")
        # get the video features
        patch_h, patch_w = H // 14, W // 14
        patch_size = (patch_h,  patch_w)
        # resize and get the video features
        x = x.view(B * T, C, H, W)
        # operate the temporal encoding
        return patch_size, x

    def forward_stream(
            self,
            video: torch.Tensor,
            queries: torch.Tensor = None,
            T_org: int = None,
            depth: torch.Tensor|np.ndarray|str=None,
            unc_metric_in: torch.Tensor|np.ndarray|str=None,
            intrs: torch.Tensor|np.ndarray|str=None,
            extrs: torch.Tensor|np.ndarray|str=None,
            queries_3d: torch.Tensor = None,
            window_len: int = 16,
            overlap_len: int = 4,
            full_point: bool = False,
            track2d_gt: torch.Tensor = None,
            fixed_cam: bool = False,
            query_no_BA: bool = False,
            stage: int = 0,
            support_frame: int = 0,
            replace_ratio: float = 0.6,
            annots_train: Dict = None,
            iters_track=4,
            **kwargs,
    ):  
        # step 1 allocate the query points on the grid
        T, C, H, W = video.shape

        if annots_train is not None:
            vis_gt = annots_train["vis"]
            _, _, N = vis_gt.shape
            number_visible = vis_gt.sum(dim=1)
            ratio_rand = torch.rand(1, N, device=vis_gt.device)
            first_positive_inds = get_nth_visible_time_index(vis_gt, (number_visible*ratio_rand).long().clamp(min=1, max=T))
            assert (torch.gather(vis_gt, 1, first_positive_inds[:, None, :].repeat(1, T, 1)) < 0).sum() == 0

            first_positive_inds = first_positive_inds.long()
            gather = torch.gather(
                annots_train["traj_3d"][...,:2], 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 2)
                )
            xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)
            queries = torch.cat([first_positive_inds[:, :, None], xys], dim=-1)[0].cpu().numpy()


        # Unfold video into segments of window_len with overlap_len
        step_slide = window_len - overlap_len
        if T < window_len:
            video_unf = video.unsqueeze(0)
            if depth is not None:
                depth_unf = depth.unsqueeze(0)
            else:
                depth_unf = None
            if unc_metric_in is not None:
                unc_metric_unf = unc_metric_in.unsqueeze(0)
            else:
                unc_metric_unf = None
            if intrs is not None:
                intrs_unf = intrs.unsqueeze(0)
            else:
                intrs_unf = None
            if extrs is not None:
                extrs_unf = extrs.unsqueeze(0)
            else:
                extrs_unf = None
        else:
            video_unf = video.unfold(0, window_len, step_slide).permute(0, 4, 1, 2, 3)  # [B, S, C, H, W]
            if depth is not None:
                depth_unf = depth.unfold(0, window_len, step_slide).permute(0, 3, 1, 2)
                intrs_unf = intrs.unfold(0, window_len, step_slide).permute(0, 3, 1, 2)
            else:
                depth_unf = None
                intrs_unf = None
            if extrs is not None:
                extrs_unf = extrs.unfold(0, window_len, step_slide).permute(0, 3, 1, 2)
            else:
                extrs_unf = None
            if unc_metric_in is not None:
                unc_metric_unf = unc_metric_in.unfold(0, window_len, step_slide).permute(0, 3, 1, 2)
            else:
                unc_metric_unf = None
        
        # parallel
        # Get number of segments
        B = video_unf.shape[0]
        #TODO: Process each segment in parallel using torch.nn.DataParallel
        c2w_traj = torch.eye(4, 4)[None].repeat(T, 1, 1)
        intrs_out = torch.eye(3, 3)[None].repeat(T, 1, 1)
        point_map = torch.zeros(T, 3, H, W).cuda()
        unc_metric = torch.zeros(T, H, W).cuda()
        # set the queries
        N, _ = queries.shape
        track3d_pred = torch.zeros(T, N, 6).cuda()
        track2d_pred = torch.zeros(T, N, 3).cuda()
        vis_pred = torch.zeros(T, N, 1).cuda()
        conf_pred = torch.zeros(T, N, 1).cuda()
        dyn_preds = torch.zeros(T, N, 1).cuda()
        # sort the queries by time
        sorted_indices = np.argsort(queries[...,0])
        sorted_inv_indices = np.argsort(sorted_indices)
        sort_query = queries[sorted_indices]
        sort_query = torch.from_numpy(sort_query).cuda()
        if queries_3d is not None:
            sort_query_3d = queries_3d[sorted_indices]
            sort_query_3d = torch.from_numpy(sort_query_3d).cuda()
        
        queries_len = 0
        overlap_d = None
        cache = None
        loss = 0.0

        for i in range(B):
            segment = video_unf[i:i+1].cuda()
            # Forward pass through model
            # detect the key points for each frames
                            
            queries_new_mask = (sort_query[...,0] < i * step_slide + window_len) * (sort_query[...,0] >= (i * step_slide + overlap_len if i > 0 else 0))
            if queries_3d is not None:
                queries_new_3d = sort_query_3d[queries_new_mask]
                queries_new_3d = queries_new_3d.float()
            else:
                queries_new_3d = None
            queries_new = sort_query[queries_new_mask.bool()]
            queries_new = queries_new.float()
            if i > 0:
                overlap2d = track2d_pred[i*step_slide:(i+1)*step_slide, :queries_len, :]
                overlapvis = vis_pred[i*step_slide:(i+1)*step_slide, :queries_len, :]
                overlapconf = conf_pred[i*step_slide:(i+1)*step_slide, :queries_len, :]
                overlap_query = (overlapvis * overlapconf).max(dim=0)[1][None, ...]
                overlap_xy = torch.gather(overlap2d, 0, overlap_query.repeat(1,1,2))
                overlap_d = torch.gather(overlap2d, 0, overlap_query.repeat(1,1,3))[...,2].detach()
                overlap_query = torch.cat([overlap_query[...,:1], overlap_xy], dim=-1)[0]
                queries_new[...,0] -= i*step_slide
                queries_new = torch.cat([overlap_query, queries_new], dim=0).detach()
            
            if annots_train is None:
                annots = {}
            else:
                annots = copy.deepcopy(annots_train)
                annots["traj_3d"] = annots["traj_3d"][:, i*step_slide:i*step_slide+window_len, sorted_indices,:][...,:len(queries_new),:]
                annots["vis"] = annots["vis"][:, i*step_slide:i*step_slide+window_len, sorted_indices][...,:len(queries_new)]
                annots["poses_gt"] =  annots["poses_gt"][:, i*step_slide:i*step_slide+window_len]
                annots["depth_gt"] = annots["depth_gt"][:, i*step_slide:i*step_slide+window_len]
                annots["intrs"] = annots["intrs"][:, i*step_slide:i*step_slide+window_len]
                annots["traj_mat"] = annots["traj_mat"][:,i*step_slide:i*step_slide+window_len]
            
            if depth is not None:
                annots["depth_gt"] = depth_unf[i:i+1].to(segment.device).to(segment.dtype)
            if unc_metric_in is not None:
                annots["unc_metric"] = unc_metric_unf[i:i+1].to(segment.device).to(segment.dtype)
            if intrs is not None:
                intr_seg = intrs_unf[i:i+1].to(segment.device).to(segment.dtype)[0].clone()
                focal = (intr_seg[:,0,0] / segment.shape[-1] + intr_seg[:,1,1]/segment.shape[-2]) / 2
                pose_fake = torch.zeros(1, 8).to(depth.device).to(depth.dtype).repeat(segment.shape[1], 1)
                pose_fake[:, -1] = focal
                pose_fake[:,3]=1
                annots["intrs_gt"] = intr_seg
            if extrs is not None:
                extrs_unf_norm = extrs_unf[i:i+1][0].clone()
                extrs_unf_norm = torch.inverse(extrs_unf_norm[:1,...]) @ extrs_unf[i:i+1][0]
                rot_vec = matrix_to_quaternion(extrs_unf_norm[:,:3,:3])
                annots["poses_gt"] = torch.zeros(1, rot_vec.shape[0], 7).to(segment.device).to(segment.dtype)
                annots["poses_gt"][:, :, 3:7] = rot_vec.to(segment.device).to(segment.dtype)[None]
                annots["poses_gt"][:, :, :3] = extrs_unf_norm[:,:3,3].to(segment.device).to(segment.dtype)[None]
                annots["use_extr"] = True
            
            kwargs.update({"stage": stage})
            
            #TODO: DEBUG
            out = self.forward(segment, pts_q=queries_new,
                                pts_q_3d=queries_new_3d, overlap_d=overlap_d,
                                full_point=full_point,
                                fixed_cam=fixed_cam, query_no_BA=query_no_BA,
                                support_frame=segment.shape[1]-1,
                                cache=cache, replace_ratio=replace_ratio,
                                iters_track=iters_track,
                                **kwargs, annots=annots)
            if self.training:
                loss += out["loss"].squeeze()

            queries_len = len(queries_new)
            # update the track3d and track2d
            left_len = len(track3d_pred[i*step_slide:i*step_slide+window_len, :queries_len, :])
            track3d_pred[i*step_slide:i*step_slide+window_len, :queries_len, :] = out["rgb_tracks"][0,:left_len,:queries_len,:]
            track2d_pred[i*step_slide:i*step_slide+window_len, :queries_len, :] = out["traj_est"][0,:left_len,:queries_len,:3]
            vis_pred[i*step_slide:i*step_slide+window_len, :queries_len, :] = out["vis_est"][0,:left_len,:queries_len,None]
            conf_pred[i*step_slide:i*step_slide+window_len, :queries_len, :] = out["conf_pred"][0,:left_len,:queries_len,None]
            dyn_preds[i*step_slide:i*step_slide+window_len, :queries_len, :] = out["dyn_preds"][0,:left_len,:queries_len,None]

            # process the output for each segment   
            seg_c2w = out["poses_pred"][0]
            seg_intrs = out["intrs"][0]
            seg_point_map = out["points_map"]
            seg_conf_depth = out["unc_metric"]
            
            # cache management
            cache = out["cache"]
            for k in cache.keys():
                if "_pyramid" in k:
                    for j in range(len(cache[k])):
                        if len(cache[k][j].shape) == 5:
                            cache[k][j] = cache[k][j][:,:,:,:queries_len,:]
                        elif len(cache[k][j].shape) == 4:
                            cache[k][j] = cache[k][j][:,:1,:queries_len,:]
                elif "_pred_cache" in k:
                    cache[k] = cache[k][-overlap_len:,:queries_len,:]
                else:
                    cache[k] = cache[k][-overlap_len:]
            
            # update the results
            idx_glob = i * step_slide
            # refine part
            # mask_update = sort_query[..., 0] < i * step_slide + window_len
            # sort_query_pick = sort_query[mask_update]
            intrs_out[idx_glob:idx_glob+window_len] = seg_intrs
            point_map[idx_glob:idx_glob+window_len] = seg_point_map
            unc_metric[idx_glob:idx_glob+window_len] = seg_conf_depth
            # update the camera poses
            
            # if using the ground truth pose
            # if extrs_unf is not None:
            #     c2w_traj[idx_glob:idx_glob+window_len] = extrs_unf[i:i+1][0].to(c2w_traj.device).to(c2w_traj.dtype)
            # else:
            prev_c2w = c2w_traj[idx_glob:idx_glob+window_len][:1]
            c2w_traj[idx_glob:idx_glob+window_len] = prev_c2w@seg_c2w.to(c2w_traj.device).to(c2w_traj.dtype)

        track2d_pred = track2d_pred[:T_org,sorted_inv_indices,:]
        track3d_pred = track3d_pred[:T_org,sorted_inv_indices,:]
        vis_pred = vis_pred[:T_org,sorted_inv_indices,:]
        conf_pred = conf_pred[:T_org,sorted_inv_indices,:]
        dyn_preds = dyn_preds[:T_org,sorted_inv_indices,:]
        unc_metric = unc_metric[:T_org,:]
        point_map = point_map[:T_org,:]
        intrs_out = intrs_out[:T_org,:]
        c2w_traj = c2w_traj[:T_org,:]
        if self.training:
            ret = {
                "loss": loss,
                "depth_loss": 0.0,
                "ab_loss": 0.0,
                "vis_loss": out["vis_loss"],
                "track_loss": out["track_loss"],
                "conf_loss": out["conf_loss"],
                "dyn_loss": out["dyn_loss"],
                "sync_loss": out["sync_loss"],
                "poses_pred": c2w_traj[None],
                "intrs": intrs_out[None],
                "points_map": point_map,
                "track3d_pred": track3d_pred[None],
                "rgb_tracks": track3d_pred[None],
                "track2d_pred": track2d_pred[None],
                "traj_est": track2d_pred[None],
                "vis_est": vis_pred[None], "conf_pred": conf_pred[None],
                "dyn_preds": dyn_preds[None],
                "imgs_raw": video[None],
                "unc_metric": unc_metric,
                }
            
            return ret    
        else:
            return c2w_traj, intrs_out, point_map, unc_metric, track3d_pred, track2d_pred, vis_pred, conf_pred
    def forward(self,
                 x: torch.Tensor,
                 annots: Dict = {},
                 pts_q: torch.Tensor = None,
                 pts_q_3d: torch.Tensor = None,
                 overlap_d: torch.Tensor = None,
                 full_point = False,
                 fixed_cam = False,
                 support_frame = 0,
                 query_no_BA = False,
                 cache = None,
                 replace_ratio = 0.6,
                 iters_track=4,
                 **kwargs):
        """
        forward the video camera model, which predict (
            `intr` `camera poses` `video depth`
            )

        args:
            x: the input video frames. [B, T, C, H, W]
            annots: the annotations for video frames.
                    {
                        "poses_gt": the pose encoding for the video frames. [B, T, 7]
                        "depth_gt": the ground truth depth for the video frames. [B, T, 1, H, W],
                        "metric": bool, whether to calculate the metric for the video frames.
                    }
        """
        self.support_frame = support_frame

        #TODO: to adjust a little bit
        track_loss=ab_loss=vis_loss=track_loss=conf_loss=dyn_loss=0.0
        B, T, _, H, W = x.shape
        imgs_raw = x.clone()
        # get the video split and features for each segment
        patch_size, x_resize = self.ProcVid(x)
        x_resize = rearrange(x_resize, "(b t) c h w -> b t c h w", b=B)
        H_resize, W_resize = x_resize.shape[-2:]
        
        prec_fx = W / W_resize
        prec_fy = H / H_resize
        # get patch size
        P_H, P_W = patch_size

        # get the depth, pointmap and mask
        #TODO: Release DepthAnything Version 
        points_map_gt = None
        with torch.no_grad():
            if_gt_depth = (("depth_gt" in annots.keys())) and (kwargs.get('stage', 0)==1 or kwargs.get('stage', 0)==3)
            if if_gt_depth==False:
                if cache is not None:
                    T_cache = cache["points_map"].shape[0]
                    T_new = T - T_cache
                    x_resize_new = x_resize[:, T_cache:]
                else:
                    T_new = T
                    x_resize_new = x_resize
                # infer with chunk
                chunk_size = self.chunk_size
                metric_depth = []
                intrs = []
                unc_metric = []
                mask = []
                points_map = []
                normals = []
                normals_mask = []
                for i in range(0, B*T_new, chunk_size):
                    output = self.base_model.infer(x_resize_new.view(B*T_new, -1, H_resize, W_resize)[i:i+chunk_size])
                    metric_depth.append(output['depth'])
                    intrs.append(output['intrinsics'])
                    unc_metric.append(output['mask_prob'])
                    mask.append(output['mask'])
                    points_map.append(output['points'])
                    normals_i, normals_mask_i = utils3d.torch.points_to_normals(output['points'], mask=output['mask'])
                    normals.append(normals_i)
                    normals_mask.append(normals_mask_i)

                metric_depth = torch.cat(metric_depth, dim=0).view(B*T_new, 1, H_resize, W_resize).to(x.dtype)
                intrs = torch.cat(intrs, dim=0).view(B, T_new, 3, 3).to(x.dtype)
                intrs[:,:,0,:] *= W_resize
                intrs[:,:,1,:] *= H_resize                
                # points_map = torch.cat(points_map, dim=0)
                mask = torch.cat(mask, dim=0).view(B*T_new, 1, H_resize, W_resize).to(x.dtype)
                # cat the normals
                normals = torch.cat(normals, dim=0)
                normals_mask = torch.cat(normals_mask, dim=0)
                
                metric_depth = metric_depth.clone()
                metric_depth[metric_depth == torch.inf] = 0
                _depths = metric_depth[metric_depth > 0].reshape(-1)
                q25 = torch.kthvalue(_depths, int(0.25 * len(_depths))).values
                q75 = torch.kthvalue(_depths, int(0.75 * len(_depths))).values
                iqr = q75 - q25
                upper_bound = (q75 + 0.8*iqr).clamp(min=1e-6, max=10*q25)
                _depth_roi = torch.tensor(
                    [1e-1, upper_bound.item()], 
                    dtype=metric_depth.dtype, 
                    device=metric_depth.device
                )
                mask_roi = (metric_depth > _depth_roi[0]) & (metric_depth < _depth_roi[1])
                mask = mask * mask_roi
                mask = mask * (~(utils3d.torch.depth_edge(metric_depth, rtol=0.03, mask=mask.bool()))) * normals_mask[:,None,...]
                points_map = depth_to_points_colmap(metric_depth.squeeze(1), intrs.view(B*T_new, 3, 3))
                unc_metric = torch.cat(unc_metric, dim=0).view(B*T_new, 1, H_resize, W_resize).to(x.dtype)
                unc_metric *= mask
                if full_point:
                    unc_metric = (~(utils3d.torch.depth_edge(metric_depth, rtol=0.1, mask=torch.ones_like(metric_depth).bool()))).float() * (metric_depth != 0)
                if cache is not None:
                    assert B==1, "only support batch size 1 right now."
                    unc_metric = torch.cat([cache["unc_metric"], unc_metric], dim=0)
                    intrs = torch.cat([cache["intrs"][None], intrs], dim=1)
                    points_map = torch.cat([cache["points_map"].permute(0,2,3,1), points_map], dim=0)
                    metric_depth = torch.cat([cache["metric_depth"], metric_depth], dim=0)
            
            if "poses_gt" in annots.keys():
                intrs, c2w_traj_gt = pose_enc2mat(annots["poses_gt"],
                                            H_resize, W_resize, self.resolution)
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
                # if (upper_bound > 200).any():
                #     import pdb; pdb.set_trace()
                if (kwargs.get('stage', 0) == 2):
                    unc_metric = ((metric_depth_gt > 0)*(mask_roi) * (unc_metric > 0.5)).float()
                    metric_depth_gt[metric_depth_gt > 10*q25] = 0
                else:
                    unc_metric = ((metric_depth_gt > 0)*(mask_roi)).float()
                    unc_metric *= (~(utils3d.torch.depth_edge(metric_depth_gt, rtol=0.03, mask=mask_roi.bool()))).float()
                    # filter the sky
                    metric_depth_gt[metric_depth_gt > 10*q25] = 0
                if "unc_metric" in annots.keys():
                    unc_metric_ = F.interpolate(annots["unc_metric"].permute(1,0,2,3), 
                                size=(H_resize, W_resize), mode='nearest')
                    unc_metric = unc_metric * unc_metric_
                if if_gt_depth:
                    points_map = depth_to_points_colmap(metric_depth_gt.squeeze(1), intrs.view(B*T, 3, 3))
                    metric_depth = metric_depth_gt
                    points_map_gt = points_map
                else:
                    points_map_gt = depth_to_points_colmap(metric_depth_gt.squeeze(1), intrs.view(B*T, 3, 3))
        
        # track the 3d points 
        ret_track = None
        regular_track = True
        dyn_preds, final_tracks = None, None
        
        if "use_extr" in annots.keys():
            init_pose = True
        else:
            init_pose = False
        # set the custom vid and valid only
        custom_vid = annots.get("custom_vid", False)
        valid_only = annots.get("data_dir", [""])[0] == "stereo4d"
        if self.training:
            if (annots["vis"].sum() > 0) and (kwargs.get('stage', 0)==1 or kwargs.get('stage', 0)==3):
                traj3d = annots['traj_3d']
                if (kwargs.get('stage', 0)==1) and (annots.get("custom_vid", False)==False):
                    support_pts_q = get_track_points(H_resize, W_resize,
                                                    T, x.device, query_size=self.track_num // 2,
                                                    support_frame=self.support_frame, unc_metric=unc_metric, mode="incremental")[None]
                else:
                    support_pts_q = get_track_points(H_resize, W_resize,
                                                    T, x.device, query_size=random.randint(32, 256),
                                                    support_frame=self.support_frame, unc_metric=unc_metric, mode="incremental")[None]
                if pts_q is not None:
                    pts_q = pts_q[None,None]
                    ret_track, dyn_preds, final_tracks, rgb_tracks, intrs_org, point_map_org_refined, cache = self.Track3D(imgs_raw,
                                                        metric_depth, 
                                                        unc_metric.detach(), points_map, pts_q,         
                                                        intrs=intrs.clone(), cache=cache,
                                                        prec_fx=prec_fx, prec_fy=prec_fy, overlap_d=overlap_d, 
                                                        vis_gt=annots['vis'], traj3d_gt=traj3d, iters=iters_track,
                                                        cam_gt=c2w_traj_gt, support_pts_q=support_pts_q, custom_vid=custom_vid,
                                                        init_pose=init_pose, fixed_cam=fixed_cam, stage=kwargs.get('stage', 0),
                                                        points_map_gt=points_map_gt, valid_only=valid_only, replace_ratio=replace_ratio)
                else:
                    ret_track, dyn_preds, final_tracks, rgb_tracks, intrs_org, point_map_org_refined, cache = self.Track3D(imgs_raw,
                                                        metric_depth, 
                                                        unc_metric.detach(), points_map, traj3d[..., :2],         
                                                        intrs=intrs.clone(), cache=cache,
                                                        prec_fx=prec_fx, prec_fy=prec_fy, overlap_d=overlap_d, 
                                                        vis_gt=annots['vis'], traj3d_gt=traj3d, iters=iters_track,
                                                        cam_gt=c2w_traj_gt, support_pts_q=support_pts_q, custom_vid=custom_vid,
                                                        init_pose=init_pose, fixed_cam=fixed_cam, stage=kwargs.get('stage', 0),
                                                        points_map_gt=points_map_gt, valid_only=valid_only, replace_ratio=replace_ratio)
                regular_track = False

                    
        if regular_track:
            if pts_q is None:
                pts_q = get_track_points(H_resize, W_resize,
                                            T, x.device, query_size=self.track_num,
                                            support_frame=self.support_frame, unc_metric=unc_metric, mode="incremental" if self.training else "incremental")[None]
                support_pts_q = None
            else:
                pts_q = pts_q[None,None]
                # resize the query points
                pts_q[...,1] *= W_resize / W
                pts_q[...,2] *= H_resize / H

                if pts_q_3d is not None:
                    pts_q_3d = pts_q_3d[None,None]
                    # resize the query points
                    pts_q_3d[...,1] *= W_resize / W
                    pts_q_3d[...,2] *= H_resize / H
                else:
                    # adjust the query with uncertainty
                    if (full_point==False) and (overlap_d is None):
                        pts_q_unc = sample_features5d(unc_metric[None], pts_q).squeeze()
                        pts_q = pts_q[:,:,pts_q_unc>0.5,:]
                        if (pts_q_unc<0.5).sum() > 0:
                            # pad the query points
                            pad_num = pts_q_unc.shape[0] - pts_q.shape[2]
                            # pick the random indices
                            indices = torch.randint(0, pts_q.shape[2], (pad_num,), device=pts_q.device)
                            pad_pts = indices
                            pts_q = torch.cat([pts_q, pts_q[:,:,pad_pts,:]], dim=-2)

                support_pts_q = get_track_points(H_resize, W_resize,
                                            T, x.device, query_size=self.track_num,
                                            support_frame=self.support_frame,
                                            unc_metric=unc_metric, mode="mixed")[None]

            points_map[points_map>1e3] = 0
            points_map = depth_to_points_colmap(metric_depth.squeeze(1), intrs.view(B*T, 3, 3))
            ret_track, dyn_preds, final_tracks, rgb_tracks, intrs_org, point_map_org_refined, cache = self.Track3D(imgs_raw,
                                                    metric_depth,
                                                    unc_metric.detach(), points_map, pts_q, 
                                                    pts_q_3d=pts_q_3d, intrs=intrs.clone(),cache=cache,
                                                    overlap_d=overlap_d, cam_gt=c2w_traj_gt if kwargs.get('stage', 0)==1 else None,
                                                    prec_fx=prec_fx, prec_fy=prec_fy, support_pts_q=support_pts_q, custom_vid=custom_vid, valid_only=valid_only,
                                                    fixed_cam=fixed_cam, query_no_BA=query_no_BA, init_pose=init_pose, iters=iters_track,
                                                    stage=kwargs.get('stage', 0), points_map_gt=points_map_gt, replace_ratio=replace_ratio)
        intrs = intrs_org
        points_map = point_map_org_refined
        c2w_traj = ret_track["cam_pred"]
        
        if ret_track is not None:
            if ret_track["loss"] is not None:
                track_loss, conf_loss, dyn_loss, vis_loss, point_map_loss, scale_loss, shift_loss, sync_loss= ret_track["loss"]

        # update the cache
        cache.update({"metric_depth": metric_depth, "unc_metric": unc_metric, "points_map": points_map, "intrs": intrs[0]})
        # output
        depth = F.interpolate(metric_depth, 
                            size=(H, W), mode='bilinear', align_corners=True).squeeze(1)
        points_map = F.interpolate(points_map, 
                            size=(H, W), mode='bilinear', align_corners=True).squeeze(1)
        unc_metric = F.interpolate(unc_metric, 
                            size=(H, W), mode='bilinear', align_corners=True).squeeze(1)
        
        if self.training:
            
            loss = track_loss + conf_loss + dyn_loss + sync_loss + vis_loss + point_map_loss + (scale_loss + shift_loss)*50
            ret = {"loss": loss,
                    "depth_loss": point_map_loss, 
                    "ab_loss": (scale_loss + shift_loss)*50,
                    "vis_loss": vis_loss, "track_loss": track_loss,
                    "poses_pred": c2w_traj, "dyn_preds": dyn_preds, "traj_est": final_tracks, "conf_loss": conf_loss,
                    "imgs_raw": imgs_raw, "rgb_tracks": rgb_tracks, "vis_est": ret_track['vis_pred'],
                    "depth": depth, "points_map": points_map, "unc_metric": unc_metric, "intrs": intrs, "dyn_loss": dyn_loss, 
                    "sync_loss": sync_loss, "conf_pred": ret_track['conf_pred'], "cache": cache,
                    }
            
        else:

            if ret_track is not None:
                traj_est = ret_track['preds']
                traj_est[..., 0] *= W / W_resize
                traj_est[..., 1] *= H / H_resize
                vis_est = ret_track['vis_pred']
            else:
                traj_est = torch.zeros(B, self.track_num // 2, 3).to(x.device)
                vis_est = torch.zeros(B, self.track_num // 2).to(x.device)

            if intrs is not None:
                intrs[..., 0, :] *= W / W_resize
                intrs[..., 1, :] *= H / H_resize
            ret = {"poses_pred": c2w_traj, "dyn_preds": dyn_preds,
                    "depth": depth, "traj_est": traj_est, "vis_est": vis_est, "imgs_raw": imgs_raw,
                    "rgb_tracks": rgb_tracks, "intrs": intrs, "unc_metric": unc_metric, "points_map": points_map,
                    "conf_pred": ret_track['conf_pred'], "cache": cache,
                    }
                
        return ret
