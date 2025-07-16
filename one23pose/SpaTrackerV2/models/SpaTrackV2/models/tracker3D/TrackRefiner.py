import os, sys
import torch
import torch.amp
from models.SpaTrackV2.models.tracker3D.co_tracker.cotracker_base import CoTrackerThreeOffline, get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.utils.model_utils import sample_features5d
from models.SpaTrackV2.models.blocks import bilinear_sampler
import torch.nn as nn
from models.SpaTrackV2.models.tracker3D.co_tracker.utils import (
    EfficientUpdateFormer, AttnBlock, Attention, CrossAttnBlock,
    sequence_BCE_loss, sequence_loss, sequence_prob_loss, sequence_dyn_prob_loss, sequence_loss_xyz, balanced_binary_cross_entropy
)
from torchvision.io import write_video
import math
from models.SpaTrackV2.models.tracker3D.co_tracker.utils import (
    Mlp, BasicEncoder, EfficientUpdateFormer, GeometryEncoder, NeighborTransformer, CorrPointformer
)
from models.SpaTrackV2.utils.embeddings import get_3d_sincos_pos_embed_from_grid
from einops import rearrange, repeat
from models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import (
    EfficientUpdateFormer3D, weighted_procrustes_torch, posenc, key_fr_wprocrustes, get_topo_mask,
    TrackFusion, get_nth_visible_time_index
)
from models.SpaTrackV2.models.tracker3D.spatrack_modules.ba import extract_static_from_3DTracks, ba_pycolmap
from models.SpaTrackV2.models.tracker3D.spatrack_modules.pointmap_updator import PointMapUpdator
from models.SpaTrackV2.models.tracker3D.spatrack_modules.alignment import affine_invariant_global_loss
from models.SpaTrackV2.models.tracker3D.delta_utils.upsample_transformer import UpsampleTransformerAlibi

class TrackRefiner3D(CoTrackerThreeOffline):

    def __init__(self, args=None):
        super().__init__(**args["base"])
        
        """
        This is 3D warpper from cotracker, which load the cotracker pretrain and
        jointly refine the `camera pose`, `3D tracks`, `video depth`, `visibility` and `conf`
        """
        self.updateformer3D = EfficientUpdateFormer3D(self.updateformer)
        self.corr_depth_mlp = Mlp(in_features=256, hidden_features=256, out_features=256)
        self.rel_pos_mlp = Mlp(in_features=75, hidden_features=128, out_features=128)
        self.rel_pos_glob_mlp = Mlp(in_features=75, hidden_features=128, out_features=256)
        self.corr_xyz_mlp = Mlp(in_features=256, hidden_features=128, out_features=128)
        self.xyz_mlp = Mlp(in_features=126, hidden_features=128, out_features=84)
        # self.track_feat_mlp = Mlp(in_features=1110, hidden_features=128, out_features=128)
        self.proj_xyz_embed = Mlp(in_features=1210+50, hidden_features=1110, out_features=1110)
        # get the anchor point's embedding, and init the pts refiner
        update_pts = True

        self.corr_transformer = nn.ModuleList([
            CorrPointformer(
                dim=128,
                num_heads=8,
                head_dim=128 // 8,
                mlp_ratio=4.0,
            )   
        ]
        )
        self.fnet = BasicEncoder(input_dim=3, 
                                 output_dim=self.latent_dim, stride=self.stride)
        self.corr3d_radius = 3
        
        self.mode = args["mode"]
        if self.mode == "online":
            self.s_wind = args["s_wind"]
            self.overlap = args["overlap"]

    def upsample_with_mask(
        self, inp: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Upsample flow field [H/P, W/P, 2] -> [H, W, 2] using convex combination"""
        H, W = inp.shape[-2:]
        up_inp = F.unfold(
            inp, [self.upsample_kernel_size, self.upsample_kernel_size], padding=(self.upsample_kernel_size - 1) // 2
        )
        up_inp = rearrange(up_inp, "b c (h w) -> b c h w", h=H, w=W)
        up_inp = F.interpolate(up_inp, scale_factor=self.upsample_factor, mode="nearest")
        up_inp = rearrange(
            up_inp, "b (c i j) h w -> b c (i j) h w", i=self.upsample_kernel_size, j=self.upsample_kernel_size
        )

        up_inp = torch.sum(mask * up_inp, dim=2)
        return up_inp

    def track_from_cam(self, queries, c2w_traj, intrs,
                                rgbs=None, visualize=False):
        """
        This function will generate tracks by camera transform

        Args:
            queries: B T N 4
            c2w_traj: B T 4 4
            intrs: B T 3 3
        """
        B, T, N, _ = queries.shape
        query_t = queries[:,0,:,0].to(torch.int64) # B N
        query_c2w = torch.gather(c2w_traj,
                                  dim=1, index=query_t[..., None, None].expand(-1, -1, 4, 4))  # B N 4 4
        query_intr = torch.gather(intrs,
                                  dim=1, index=query_t[..., None, None].expand(-1, -1, 3, 3))  # B N 3 3
        query_pts = queries[:,0,:,1:4].clone() # B N 3
        query_d = queries[:,0,:,3:4] # B N 3
        query_pts[...,2] = 1

        cam_pts = torch.einsum("bnij,bnj->bni", torch.inverse(query_intr), query_pts)*query_d # B N 3
        # convert to world
        cam_pts_h = torch.zeros(B, N, 4, device=cam_pts.device)
        cam_pts_h[..., :3] = cam_pts 
        cam_pts_h[..., 3] = 1 
        world_pts = torch.einsum("bnij,bnj->bni", query_c2w, cam_pts_h)
        # convert to other frames
        cam_other_pts_ = torch.einsum("btnij,btnj->btni", 
                                    torch.inverse(c2w_traj[:,:,None].float().repeat(1,1,N,1,1)),
                                    world_pts[:,None].repeat(1,T,1,1))
        cam_depth = cam_other_pts_[...,2:3]
        cam_other_pts = cam_other_pts_[...,:3] / (cam_other_pts_[...,2:3].abs()+1e-6)
        cam_other_pts = torch.einsum("btnij,btnj->btni", intrs[:,:,None].repeat(1,1,N,1,1), cam_other_pts[...,:3])
        cam_other_pts[..., 2:] = cam_depth
        
        if visualize:
            viser = Visualizer(save_dir=".", grayscale=True, 
                               fps=10, pad_value=50, tracks_leave_trace=0)
            cam_other_pts[..., 0] /= self.factor_x
            cam_other_pts[..., 1] /= self.factor_y
            viser.visualize(video=rgbs, tracks=cam_other_pts[..., :2], filename="test")

        
        init_xyzs = cam_other_pts
        
        return init_xyzs, world_pts[..., :3], cam_other_pts_[..., :3]
    
    def cam_from_track(self, tracks, intrs,
                       dyn_prob=None, metric_unc=None,
                       vis_est=None, only_cam_pts=False,
                       track_feat_concat=None,
                       tracks_xyz=None,
                       query_pts=None,
                       fixed_cam=False,
                       depth_unproj=None,
                       cam_gt=None,
                       init_pose=False,
                       ):
        """
        This function will generate tracks by camera transform

        Args:
            queries: B T N 3
            scale_est: 1 1
            shift_est: 1 1
            intrs: B T 3 3
            dyn_prob: B T N
            metric_unc: B N 1
            query_pts: B T N 3
        """
        if tracks_xyz is not None:
            B, T, N, _ = tracks.shape
            cam_pts = tracks_xyz
            intr_repeat = intrs[:,:,None].repeat(1,1,N,1,1)
        else:
            B, T, N, _ = tracks.shape
            # get the pts in cam coordinate
            tracks_xy = tracks[...,:3].clone().detach() # B T N 3
            # tracks_z = 1/(tracks[...,2:] * scale_est + shift_est) # B T N 1
            tracks_z = tracks[...,2:].detach() # B T N 1
            tracks_xy[...,2] = 1
            intr_repeat = intrs[:,:,None].repeat(1,1,N,1,1)
            cam_pts = torch.einsum("bnij,bnj->bni", 
                                torch.inverse(intr_repeat.view(B*T,N,3,3)).float(),
                                    tracks_xy.view(B*T, N, 3))*(tracks_z.view(B*T,N,1).abs()) # B*T N 3
            cam_pts[...,2] *= torch.sign(tracks_z.view(B*T,N))
            # get the normalized cam pts, and pts refiner
            mask_z = (tracks_z.max(dim=1)[0]<200).squeeze()
            cam_pts = cam_pts.view(B, T, N, 3)

        if only_cam_pts:
            return cam_pts
        dyn_prob = dyn_prob.mean(dim=1)[..., None]
        # B T N 3 -> local frames coordinates.  transformer  static points  B T N 3 -> B T N 3  static (B T N 3) -> same -> dynamic points @ C2T.inverse()
        # get the cam pose
        vis_est_ = vis_est[:,:,None,:]
        graph_matrix = (vis_est_*vis_est_.permute(0, 2,1,3)).detach()
        # find the max connected component
        key_fr_idx = [0]
        weight_final = (metric_unc) # * vis_est


        with torch.amp.autocast(enabled=False, device_type='cuda'):
            if fixed_cam:
                c2w_traj_init = self.c2w_est_curr
                c2w_traj_glob = c2w_traj_init
                cam_pts_refine = cam_pts
                intrs_refine = intrs
                xy_refine = query_pts[...,1:3]
                world_tracks_init = torch.einsum("btij,btnj->btni", c2w_traj_init[:,:,:3,:3], cam_pts) + c2w_traj_init[:,:,None,:3,3] 
                world_tracks_refined = world_tracks_init
                # extract the stable static points for refine the camera pose
                intrs_dn = intrs.clone()
                intrs_dn[...,0,:] *= self.factor_x
                intrs_dn[...,1,:] *= self.factor_y
                _, query_world_pts, _ = self.track_from_cam(query_pts, c2w_traj_init, intrs_dn)
                world_tracks_static, mask_static, mask_topk, vis_mask_static, tracks2d_static = extract_static_from_3DTracks(world_tracks_init,
                                                                                                                dyn_prob, query_world_pts,
                                                                                                                vis_est, tracks, img_size=self.image_size,
                                                                                                                K=0)
                world_static_refine = world_tracks_static

            else:

                if (not self.training):
                    # if (self.c2w_est_curr==torch.eye(4, device=cam_pts.device).repeat(B, T, 1, 1)).all():
                    campts_update = torch.einsum("btij,btnj->btni", self.c2w_est_curr[...,:3,:3], cam_pts) + self.c2w_est_curr[...,None,:3,3]
                    # campts_update = cam_pts
                    c2w_traj_init_update = key_fr_wprocrustes(campts_update, graph_matrix,
                                                                (weight_final*(1-dyn_prob)).permute(0,2,1), vis_est_.permute(0,1,3,2)) 
                    c2w_traj_init = c2w_traj_init_update@self.c2w_est_curr                                   
                    # else:
                        # c2w_traj_init = self.c2w_est_curr                # extract the stable static points for refine the camera pose
                else:
                    # if (self.c2w_est_curr==torch.eye(4, device=cam_pts.device).repeat(B, T, 1, 1)).all():
                    campts_update = torch.einsum("btij,btnj->btni", self.c2w_est_curr[...,:3,:3], cam_pts) + self.c2w_est_curr[...,None,:3,3]
                    # campts_update = cam_pts
                    c2w_traj_init_update = key_fr_wprocrustes(campts_update, graph_matrix,
                                                                (weight_final*(1-dyn_prob)).permute(0,2,1), vis_est_.permute(0,1,3,2)) 
                    c2w_traj_init = c2w_traj_init_update@self.c2w_est_curr                                  
                    # else:
                        # c2w_traj_init = self.c2w_est_curr                # extract the stable static points for refine the camera pose
                
                intrs_dn = intrs.clone()
                intrs_dn[...,0,:] *= self.factor_x
                intrs_dn[...,1,:] *= self.factor_y
                _, query_world_pts, _ = self.track_from_cam(query_pts, c2w_traj_init, intrs_dn)
                # refine the world tracks
                world_tracks_init = torch.einsum("btij,btnj->btni", c2w_traj_init[:,:,:3,:3], cam_pts) + c2w_traj_init[:,:,None,:3,3] 
                world_tracks_static, mask_static, mask_topk, vis_mask_static, tracks2d_static = extract_static_from_3DTracks(world_tracks_init,
                                                                                                            dyn_prob, query_world_pts,
                                                                                                            vis_est, tracks, img_size=self.image_size,
                                                                                                            K=150 if self.training else 1500)
                # calculate the efficient ba
                cam_tracks_static = cam_pts[:,:,mask_static.squeeze(),:][:,:,mask_topk.squeeze(),:]
                cam_tracks_static[...,2] = depth_unproj.view(B, T, N)[:,:,mask_static.squeeze()][:,:,mask_topk.squeeze()]

                c2w_traj_glob, world_static_refine, intrs_refine = ba_pycolmap(world_tracks_static, intrs,
                                                                                c2w_traj_init, vis_mask_static,
                                                                                tracks2d_static, self.image_size,
                                                                                cam_tracks_static=cam_tracks_static,
                                                                                training=self.training, query_pts=query_pts)
                c2w_traj_glob = c2w_traj_glob.view(B, T, 4, 4)
                world_tracks_refined = world_tracks_init
            
            #NOTE: merge the index of static points and topk points
            # merge_idx = torch.where(mask_static.squeeze()>0)[0][mask_topk.squeeze()]
            # world_tracks_refined[:,:,merge_idx] = world_static_refine
            
            # test the procrustes
            w2c_traj_glob = torch.inverse(c2w_traj_init.detach())
            cam_pts_refine = torch.einsum("btij,btnj->btni", w2c_traj_glob[:,:,:3,:3], world_tracks_refined) + w2c_traj_glob[:,:,None,:3,3]
            # get the xyz_refine
            #TODO: refiner
            cam_pts4_proj = cam_pts_refine.clone()
            cam_pts4_proj[...,2] *= torch.sign(cam_pts4_proj[...,2:3].view(B*T,N))
            xy_refine = torch.einsum("btnij,btnj->btni", intrs_refine.view(B,T,1,3,3).repeat(1,1,N,1,1), cam_pts4_proj/cam_pts4_proj[...,2:3].abs())
            xy_refine[..., 2] = cam_pts4_proj[...,2:3].view(B*T,N)
        # xy_refine = torch.zeros_like(cam_pts_refine)[...,:2]
        return c2w_traj_glob, cam_pts_refine, intrs_refine, xy_refine, world_tracks_init, world_tracks_refined, c2w_traj_init
    
    def extract_img_feat(self, video, fmaps_chunk_size=200):
        B, T, C, H, W = video.shape
        dtype = video.dtype
        H4, W4 = H // self.stride, W // self.stride
        # Compute convolutional features for the video or for the current chunk in case of online mode
        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t : t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C, H, W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:
            fmaps = self.fnet(video.reshape(-1, C, H, W))
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)

        return fmaps

    def norm_xyz(self, xyz):
        """
        xyz can be (B T N 3) or (B T 3 H W) or (B N 3)
        """
        if xyz.ndim == 3:
            min_pts = self.min_pts
            max_pts = self.max_pts
            return (xyz - min_pts[None,None,:]) / (max_pts - min_pts)[None,None,:] * 2 - 1
        elif xyz.ndim == 4:
            min_pts = self.min_pts
            max_pts = self.max_pts
            return (xyz - min_pts[None,None,None,:]) / (max_pts - min_pts)[None,None,None,:] * 2 - 1
        elif xyz.ndim == 5:
            if xyz.shape[2] == 3:   
                min_pts = self.min_pts
                max_pts = self.max_pts
                return (xyz - min_pts[None,None,:,None,None]) / (max_pts - min_pts)[None,None,:,None,None] * 2 - 1
            elif xyz.shape[-1] == 3:
                min_pts = self.min_pts
                max_pts = self.max_pts
                return (xyz - min_pts[None,None,None,None,:]) / (max_pts - min_pts)[None,None,None,None,:] * 2 - 1

    def denorm_xyz(self, xyz):
        """
        xyz can be (B T N 3) or (B T 3 H W) or (B N 3)
        """
        if xyz.ndim == 3:
            min_pts = self.min_pts
            max_pts = self.max_pts
            return (xyz + 1) / 2 * (max_pts - min_pts)[None,None,:] + min_pts[None,None,:]
        elif xyz.ndim == 4:
            min_pts = self.min_pts
            max_pts = self.max_pts
            return (xyz + 1) / 2 * (max_pts - min_pts)[None,None,None,:] + min_pts[None,None,None,:]
        elif xyz.ndim == 5:
            if xyz.shape[2] == 3:
                min_pts = self.min_pts
                max_pts = self.max_pts
                return (xyz + 1) / 2 * (max_pts - min_pts)[None,None,:,None,None] + min_pts[None,None,:,None,None]
            elif xyz.shape[-1] == 3:
                min_pts = self.min_pts
                max_pts = self.max_pts
                return (xyz + 1) / 2 * (max_pts - min_pts)[None,None,None,None,:] + min_pts[None,None,None,None,:] 

    def forward(
        self,
        video,
        metric_depth,
        metric_unc,
        point_map,
        queries,
        pts_q_3d=None,
        overlap_d=None,
        iters=4,
        add_space_attn=True,
        fmaps_chunk_size=200,
        intrs=None,
        traj3d_gt=None,
        custom_vid=False,
        vis_gt=None,
        prec_fx=None,
        prec_fy=None,
        cam_gt=None,
        init_pose=False,
        support_pts_q=None,
        update_pointmap=True,
        fixed_cam=False,
        query_no_BA=False,
        stage=0,
        cache=None,
        points_map_gt=None,
        valid_only=False,
        replace_ratio=0.6,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3 H W]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            vdp_feats_cache: last layer's feature of depth
            tracks_init: B T N 3 the initialization of 3D tracks computed by cam pose
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        self.stage = stage

        if cam_gt is not None:
            cam_gt = cam_gt.clone()
            cam_gt = torch.inverse(cam_gt[:,:1,...])@cam_gt
        B, T, C, _, _ = video.shape
        _, _, H_, W_ = metric_depth.shape
        _, _, N, __ = queries.shape
        if (vis_gt is not None)&(queries.shape[1] == T):
            aug_visb = True
            if aug_visb:
                number_visible = vis_gt.sum(dim=1)
                ratio_rand = torch.rand(B, N, device=vis_gt.device)
                # first_positive_inds = get_nth_visible_time_index(vis_gt, 1)
                first_positive_inds = get_nth_visible_time_index(vis_gt, (number_visible*ratio_rand).long().clamp(min=1, max=T))

                assert (torch.gather(vis_gt, 1, first_positive_inds[:, None, :].repeat(1, T, 1)) < 0).sum() == 0
            else:
                __, first_positive_inds = torch.max(vis_gt, dim=1)
            first_positive_inds = first_positive_inds.long()
            gather = torch.gather(
                queries, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 2)
                )
            xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)
            gather_xyz = torch.gather(
                traj3d_gt, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 3)
            )
            z_gt_query = torch.diagonal(gather_xyz, dim1=1, dim2=2).permute(0, 2, 1)[...,2]
            queries = torch.cat([first_positive_inds[:, :, None], xys], dim=-1)
            queries = torch.cat([queries, support_pts_q[:,0]], dim=1)
        else:
            # Generate the 768 points randomly in the whole video
            queries = queries.squeeze(1)
            ba_len = queries.shape[1]
            z_gt_query = None
            if support_pts_q is not None:
                queries = torch.cat([queries, support_pts_q[:,0]], dim=1)
        
        if (abs(prec_fx-1.0) > 1e-4) & (self.training) & (traj3d_gt is not None):
            traj3d_gt[..., 0] /= prec_fx
            traj3d_gt[..., 1] /= prec_fy
            queries[...,1] /= prec_fx
            queries[...,2] /= prec_fy

        video_vis = F.interpolate(video.clone().view(B*T, 3, video.shape[-2], video.shape[-1]), (H_, W_), mode="bilinear", align_corners=False).view(B, T, 3, H_, W_)

        self.image_size = torch.tensor([H_, W_])
        # self.model_resolution = (H_, W_)
        # resize the queries and intrs
        self.factor_x = self.model_resolution[1]/W_
        self.factor_y = self.model_resolution[0]/H_   
        queries[...,1] *= self.factor_x
        queries[...,2] *= self.factor_y
        intrs_org = intrs.clone()
        intrs[...,0,:] *= self.factor_x
        intrs[...,1,:] *= self.factor_y
        
        # get the fmaps and color features
        video = F.interpolate(video.view(B*T, 3, video.shape[-2], video.shape[-1]), 
                              (self.model_resolution[0], self.model_resolution[1])).view(B, T, 3, self.model_resolution[0], self.model_resolution[1])
        _, _, _, H, W = video.shape
        if cache is not None:
            T_cache = cache["fmaps"].shape[0]
            fmaps = self.extract_img_feat(video[:,T_cache:], fmaps_chunk_size=fmaps_chunk_size)
            fmaps = torch.cat([cache["fmaps"][None], fmaps], dim=1)
        else:
            fmaps = self.extract_img_feat(video, fmaps_chunk_size=fmaps_chunk_size)
        fmaps_org = fmaps.clone()
        
        metric_depth = F.interpolate(metric_depth.view(B*T, 1, H_, W_), 
                              (self.model_resolution[0], self.model_resolution[1]),mode="nearest").view(B*T, 1, self.model_resolution[0], self.model_resolution[1]).clamp(0.01, 200)
        self.metric_unc_org = metric_unc.clone()
        metric_unc = F.interpolate(metric_unc.view(B*T, 1, H_, W_),
                                (self.model_resolution[0], self.model_resolution[1]),mode="nearest").view(B*T, 1, self.model_resolution[0], self.model_resolution[1])
        if (self.stage == 2) & (self.training):
            scale_rand = (torch.rand(B, T, device=video.device) - 0.5) + 1
            point_map = scale_rand.view(B*T,1,1,1) * point_map
        
        point_map_org = point_map.permute(0,3,1,2).view(B*T, 3, H_, W_).clone()
        point_map = F.interpolate(point_map_org.clone(),
                                  (self.model_resolution[0], self.model_resolution[1]),mode="nearest").view(B*T, 3, self.model_resolution[0], self.model_resolution[1])
        # align the point map
        point_map_org_train = point_map_org.view(B*T, 3, H_, W_).clone()

        if (stage == 2):
            # align the point map
            try:
                self.pred_points, scale_gt, shift_gt = affine_invariant_global_loss(
                    point_map_org_train.permute(0,2,3,1), 
                    points_map_gt, 
                    mask=self.metric_unc_org[:,0]>0.5,
                    align_resolution=32,
                    only_align=True
                )
            except:
                scale_gt, shift_gt = torch.ones(B*T).to(video.device), torch.zeros(B*T,3).to(video.device)
            self.scale_gt, self.shift_gt = scale_gt, shift_gt
        else:
            scale_est, shift_est = None, None

        # extract the pts features
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0

        B, N, __ = queries.shape
        queries_z = sample_features5d(metric_depth.view(B, T, 1, H, W), 
                                                queries[:,None], interp_mode="nearest").squeeze(1)
        queries_z_unc = sample_features5d(metric_unc.view(B, T, 1, H, W),
                                                queries[:,None], interp_mode="nearest").squeeze(1)
        
        queries_rgb = sample_features5d(video.view(B, T, C, H, W),
                                                queries[:,None], interp_mode="nearest").squeeze(1)
        queries_point_map = sample_features5d(point_map.view(B, T, 3, H, W),
                                                    queries[:,None], interp_mode="nearest").squeeze(1)
        if ((queries_z > 100)*(queries_z == 0)).sum() > 0:
            import pdb; pdb.set_trace()
        
        if overlap_d is not None:
            queries_z[:,:overlap_d.shape[1],:] = overlap_d[...,None]
            queries_point_map[:,:overlap_d.shape[1],2:] = overlap_d[...,None]
        
        if pts_q_3d is not None:
            scale_factor = (pts_q_3d[...,-1].permute(0,2,1) / queries_z[:,:pts_q_3d.shape[2],:]).squeeze().median()
            queries_z[:,:pts_q_3d.shape[2],:] = pts_q_3d[...,-1].permute(0,2,1) / scale_factor
            queries_point_map[:,:pts_q_3d.shape[2],2:] = pts_q_3d[...,-1].permute(0,2,1) / scale_factor
        
        # normalize the points
        self.min_pts, self.max_pts = queries_point_map.mean(dim=(0,1)) - 3*queries_point_map.std(dim=(0,1)), queries_point_map.mean(dim=(0,1)) + 3*queries_point_map.std(dim=(0,1))
        queries_point_map = self.norm_xyz(queries_point_map)
        queries_point_map_ = queries_point_map.reshape(B, 1, N, 3).expand(B, T, N, 3).clone()
        point_map = self.norm_xyz(point_map.view(B, T, 3, H, W)).view(B*T, 3, H, W)
        
        if z_gt_query is not None:
            queries_z[:,:z_gt_query.shape[1],:] = z_gt_query[:,:,None]
            mask_traj_gt = ((queries_z[:,:z_gt_query.shape[1],:] - z_gt_query[:,:,None])).abs() < 0.1
        else:
            if traj3d_gt is not None:
                mask_traj_gt = torch.ones_like(queries_z[:, :traj3d_gt.shape[2]]).bool()
            else:
                mask_traj_gt = torch.ones_like(queries_z).bool()
        
        queries_xyz = torch.cat([queries, queries_z], dim=-1)[:,None].repeat(1, T, 1, 1)
        if cache is not None:
            cache_T, cache_N = cache["track2d_pred_cache"].shape[0], cache["track2d_pred_cache"].shape[1]
            cachexy = cache["track2d_pred_cache"].clone()
            cachexy[...,0] = cachexy[...,0] * self.factor_x 
            cachexy[...,1] = cachexy[...,1] * self.factor_y
            # initialize the 2d points with cache
            queries_xyz[:,:cache_T,:cache_N,1:] = cachexy
            queries_xyz[:,cache_T:,:cache_N,1:] = cachexy[-1:]
            # initialize the 3d points with cache
            queries_point_map_[:,:cache_T,:cache_N,:] = self.norm_xyz(cache["track3d_pred_cache"][None])
            queries_point_map_[:,cache_T:,:cache_N,:] = self.norm_xyz(cache["track3d_pred_cache"][-1:][None])

        if cam_gt is not None:
            q_static_proj, q_xyz_world, q_xyz_cam = self.track_from_cam(queries_xyz, cam_gt,
                                intrs, rgbs=video_vis, visualize=False)
            q_static_proj[..., 0] /= self.factor_x
            q_static_proj[..., 1] /= self.factor_y
                

        assert T >= 1  # A tracker needs at least two frames to track something
        video = 2 * (video / 255.0) - 1.0
        dtype = video.dtype
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride

        # We store our predictions here
        (all_coords_predictions, all_coords_xyz_predictions,all_vis_predictions,
         all_confidence_predictions, all_cam_predictions, all_dynamic_prob_predictions,
         all_cam_pts_predictions, all_world_tracks_predictions, all_world_tracks_refined_predictions,
         all_scale_est, all_shift_est) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            []
        )

        # We compute track features
        fmaps_pyramid = []
        point_map_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        track_feat3d_pyramid = []
        track_feat_support3d_pyramid = []
        track_depth_support_pyramid = []
        track_point_map_pyramid = []
        track_point_map_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        metric_depth = metric_depth
        point_map = point_map
        metric_depth_align = F.interpolate(metric_depth, scale_factor=0.25, mode='nearest')
        point_map_align = F.interpolate(point_map, scale_factor=0.25, mode='nearest')
        point_map_pyramid.append(point_map_align.view(B, T, 3, point_map_align.shape[-2], point_map_align.shape[-1]))
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)
            # downsample the depth
            metric_depth_ = metric_depth_align.reshape(B*T,1,metric_depth_align.shape[-2],metric_depth_align.shape[-1])
            metric_depth_ = F.interpolate(metric_depth_, scale_factor=0.5, mode='nearest')
            metric_depth_align = metric_depth_.reshape(B,T,1,metric_depth_.shape[-2], metric_depth_.shape[-1])
            # downsample the point map
            point_map_ = point_map_align.reshape(B*T,3,point_map_align.shape[-2],point_map_align.shape[-1])
            point_map_ = F.interpolate(point_map_, scale_factor=0.5, mode='nearest')
            point_map_align = point_map_.reshape(B,T,3,point_map_.shape[-2], point_map_.shape[-1])
            point_map_pyramid.append(point_map_align)

        for i in range(self.corr_levels):
            if cache is not None:
                cache_N = cache["track_feat_pyramid"][i].shape[2]
                track_feat_cached, track_feat_support_cached = cache["track_feat_pyramid"][i], cache["track_feat_support_pyramid"][i]
                track_feat3d_cached, track_feat_support3d_cached = cache["track_feat3d_pyramid"][i], cache["track_feat_support3d_pyramid"][i]
                track_point_map_cached, track_point_map_support_cached = self.norm_xyz(cache["track_point_map_pyramid"][i]), self.norm_xyz(cache["track_point_map_support_pyramid"][i])
                queried_coords_new = queried_coords[:,cache_N:,:] / 2**i
                queried_frames_new = queried_frames[:,cache_N:]
            else:
                queried_coords_new = queried_coords / 2**i
                queried_frames_new = queried_frames
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames_new,
                queried_coords_new,
                support_radius=self.corr_radius,
            )
            # get 3d track feat
            track_point_map, track_point_map_support = self.get_track_feat(
                point_map_pyramid[i],
                queried_frames_new,
                queried_coords_new,
                support_radius=self.corr3d_radius,
            )
            track_feat3d, track_feat_support3d = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames_new,
                queried_coords_new,
                support_radius=self.corr3d_radius,
            )
            if cache is not None:
                track_feat = torch.cat([track_feat_cached, track_feat], dim=2)
                track_point_map = torch.cat([track_point_map_cached, track_point_map], dim=2)
                track_feat_support = torch.cat([track_feat_support_cached[:,0], track_feat_support], dim=2)
                track_point_map_support = torch.cat([track_point_map_support_cached[:,0], track_point_map_support], dim=2)
                track_feat3d = torch.cat([track_feat3d_cached, track_feat3d], dim=2)
                track_feat_support3d = torch.cat([track_feat_support3d_cached[:,0], track_feat_support3d], dim=2)
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))
            track_feat3d_pyramid.append(track_feat3d.repeat(1, T, 1, 1))
            track_feat_support3d_pyramid.append(track_feat_support3d.unsqueeze(1))
            track_point_map_pyramid.append(track_point_map.repeat(1, T, 1, 1))
            track_point_map_support_pyramid.append(track_point_map_support.unsqueeze(1))

        
        D_coords = 2
        (coord_preds, coords_xyz_preds, vis_preds, confidence_preds,
         dynamic_prob_preds, cam_preds, pts3d_cam_pred, world_tracks_pred,
         world_tracks_refined_pred, point_map_preds, scale_ests, shift_ests) = (
            [], [], [], [], [], [], [], [], [], [], [], []
        )

        c2w_ests = []
        vis = torch.zeros((B, T, N), device=device).float()
        confidence = torch.zeros((B, T, N), device=device).float()
        dynamic_prob = torch.zeros((B, T, N), device=device).float()
        pro_analysis_w = torch.zeros((B, T, N), device=device).float()
        
        coords = queries_xyz[...,1:].clone()
        coords[...,:2] /= self.stride
        # coords[...,:2] = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()[...,:2]
        # initialize the 3d points
        coords_xyz = queries_point_map_.clone()
        
        # if cache is not None:
        #     viser = Visualizer(save_dir=".", grayscale=True, 
        #                        fps=10, pad_value=50, tracks_leave_trace=0)
        #     coords_clone = coords.clone()
        #     coords_clone[...,:2] *= self.stride
        #     coords_clone[..., 0] /= self.factor_x
        #     coords_clone[..., 1] /= self.factor_y
        #     viser.visualize(video=video_vis, tracks=coords_clone[..., :2], filename="test")
        #     import pdb; pdb.set_trace()

        if init_pose:
            q_init_proj, q_xyz_world, q_xyz_cam = self.track_from_cam(queries_xyz, cam_gt,
                                intrs, rgbs=video_vis, visualize=False)
            q_init_proj[..., 0] /= self.stride
            q_init_proj[..., 1] /= self.stride

        r = 2 * self.corr_radius + 1
        r_depth = 2 * self.corr3d_radius + 1
        anchor_loss = 0
        # two current states
        self.c2w_est_curr = torch.eye(4, device=device).repeat(B, T , 1, 1)
        coords_proj_curr = coords.view(B * T, N, 3)[...,:2]
        if init_pose:
            self.c2w_est_curr = cam_gt.to(coords_proj_curr.device).to(coords_proj_curr.dtype)
        sync_loss = 0
        if stage == 2:
            extra_sparse_tokens = self.scale_shift_tokens[:,:,None,:].repeat(B, 1, T, 1)
            extra_dense_tokens = self.residual_embedding[None,None].repeat(B, T, 1, 1, 1)
            xyz_pos_enc = posenc(point_map_pyramid[-2].permute(0,1,3,4,2), min_deg=0, max_deg=10).permute(0,1,4,2,3)
            extra_dense_tokens = torch.cat([xyz_pos_enc, extra_dense_tokens, fmaps_pyramid[-2]], dim=2)
            extra_dense_tokens = rearrange(extra_dense_tokens, 'b t c h w -> (b t) c h w')
            extra_dense_tokens = self.dense_mlp(extra_dense_tokens)
            extra_dense_tokens = rearrange(extra_dense_tokens, '(b t) c h w -> b t c h w', b=B, t=T)
        else:
            extra_sparse_tokens = None
            extra_dense_tokens = None
        
        scale_est, shift_est = torch.ones(B, T, 1, 1, device=device), torch.zeros(B, T, 1, 3, device=device)
        residual_point = torch.zeros(B, T, 3, self.model_resolution[0]//self.stride,
                                                         self.model_resolution[1]//self.stride, device=device)

        for it in range(iters):
            # query points scale and shift
            scale_est_query = torch.gather(scale_est, dim=1, index=queries[:,:,None,:1].long())
            shift_est_query = torch.gather(shift_est, dim=1, index=queries[:,:,None,:1].long().repeat(1, 1, 1, 3))

            coords = coords.detach()  # B T N 3
            coords_xyz = coords_xyz.detach()
            vis = vis.detach()
            confidence = confidence.detach()
            dynamic_prob = dynamic_prob.detach()
            pro_analysis_w = pro_analysis_w.detach()
            coords_init = coords.view(B * T, N, 3)
            coords_xyz_init = coords_xyz.view(B * T, N, 3)
            corr_embs = []
            corr_depth_embs = []
            corr_feats = []
            for i in range(self.corr_levels):
                # K_level = int(32*0.8**(i))
                K_level = 16
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init[...,:2] / 2**i
                )
                #NOTE: update the point map
                residual_point_i = F.interpolate(residual_point.view(B*T,3,residual_point.shape[-2],residual_point.shape[-1]),
                                                                                     size=(point_map_pyramid[i].shape[-2], point_map_pyramid[i].shape[-1]), mode='nearest')
                point_map_pyramid_i = (self.denorm_xyz(point_map_pyramid[i]) * scale_est[...,None]
                                                     + shift_est.permute(0,1,3,2)[...,None] + residual_point_i.view(B,T,3,point_map_pyramid[i].shape[-2], point_map_pyramid[i].shape[-1])).clone().detach()
                
                corr_point_map = self.get_correlation_feat(
                    self.norm_xyz(point_map_pyramid_i), coords_proj_curr / 2**i, radius=self.corr3d_radius
                )

                corr_point_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_proj_curr / 2**i, radius=self.corr3d_radius
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                track_feat_support3d = (
                    track_feat_support3d_pyramid[i]
                    .view(B, 1, r_depth, r_depth, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                #NOTE: update the point map
                track_point_map_support_pyramid_i = (self.denorm_xyz(track_point_map_support_pyramid[i]) * scale_est_query.view(B,1,1,N,1)
                                                                                                + shift_est_query.view(B,1,1,N,3)).clone().detach()

                track_point_map_support = (
                    self.norm_xyz(track_point_map_support_pyramid_i)
                    .view(B, 1, r_depth, r_depth, N, 3)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B, T, N, r * r * r * r))

                with torch.no_grad():
                    rel_pos_query_ = track_point_map_support - track_point_map_support[:,:,self.corr3d_radius,self.corr3d_radius,:][...,None,None,:]
                    rel_pos_target_ = corr_point_map - coords_xyz_init.view(B, T, N, 1, 1, 3)
                    # select the top 9 points
                    rel_pos_query_idx = rel_pos_query_.norm(dim=-1).view(B, N, -1).topk(K_level+1, dim=-1, largest=False)[1][...,1:,None]
                    rel_pos_target_idx = rel_pos_target_.norm(dim=-1).view(B, T, N, -1).topk(K_level+1, dim=-1, largest=False)[1][...,1:,None]
                    rel_pos_query_ = torch.gather(rel_pos_query_.view(B, N, -1, 3), dim=-2, index=rel_pos_query_idx.expand(B, N, K_level, 3))
                    rel_pos_target_ = torch.gather(rel_pos_target_.view(B, T, N, -1, 3), dim=-2, index=rel_pos_target_idx.expand(B, T, N, K_level, 3))
                    rel_pos_query = rel_pos_query_
                    rel_pos_target = rel_pos_target_
                    rel_pos_query = posenc(rel_pos_query, min_deg=0, max_deg=12)
                    rel_pos_target = posenc(rel_pos_target, min_deg=0, max_deg=12)
                rel_pos_target = self.rel_pos_mlp(rel_pos_target)
                rel_pos_query = self.rel_pos_mlp(rel_pos_query)
                with torch.no_grad():
                    # integrate with feature
                    track_feat_support_ = rearrange(track_feat_support3d, 'b n r k c -> b n (r k) c', r=r_depth, k=r_depth, n=N, b=B)
                    track_feat_support_ = torch.gather(track_feat_support_, dim=-2, index=rel_pos_query_idx.expand(B, N, K_level, 128))
                    queried_feat = torch.cat([rel_pos_query, track_feat_support_], dim=-1)
                    corr_feat_ = rearrange(corr_point_feat, 'b t n r k c -> b t n (r k) c', t=T, n=N, b=B)
                    corr_feat_ = torch.gather(corr_feat_, dim=-2, index=rel_pos_target_idx.expand(B, T, N, K_level, 128))
                    target_feat = torch.cat([rel_pos_target, corr_feat_], dim=-1)
                
                # 3d attention
                queried_feat = self.corr_xyz_mlp(queried_feat)
                target_feat = self.corr_xyz_mlp(target_feat)
                queried_feat = repeat(queried_feat, 'b n k c -> b t n k c', k=K_level, t=T, n=N, b=B)
                corr_depth_emb = self.corr_transformer[0](queried_feat.reshape(B*T*N,-1,128),
                                                        target_feat.reshape(B*T*N,-1,128),
                                                        target_rel_pos=rel_pos_target.reshape(B*T*N,-1,128))
                corr_depth_emb = rearrange(corr_depth_emb, '(b t n) 1 c -> b t n c', t=T, n=N, b=B)
                corr_depth_emb = self.corr_depth_mlp(corr_depth_emb)
                valid_mask = self.denorm_xyz(coords_xyz_init).view(B, T, N, -1)[...,2:3] > 0
                corr_depth_embs.append(corr_depth_emb*valid_mask)

                corr_embs.append(corr_emb)
            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, T, N, corr_embs.shape[-1])
            corr_depth_embs = torch.cat(corr_depth_embs, dim=-1)
            corr_depth_embs = corr_depth_embs.view(B, T, N, corr_depth_embs.shape[-1])
            transformer_input = [vis[..., None], confidence[..., None], corr_embs]
            transformer_input_depth = [vis[..., None], confidence[..., None], corr_depth_embs]

            rel_coords_forward = coords[:,:-1,...,:2] - coords[:,1:,...,:2]
            rel_coords_backward = coords[:, 1:,...,:2] - coords[:, :-1,...,:2]

            rel_xyz_forward = coords_xyz[:,:-1,...,:3] - coords_xyz[:,1:,...,:3]
            rel_xyz_backward = coords_xyz[:, 1:,...,:3] - coords_xyz[:, :-1,...,:3]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )
            rel_xyz_forward = torch.nn.functional.pad(
                rel_xyz_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_xyz_backward = torch.nn.functional.pad(
                rel_xyz_backward, (0, 0, 0, 0, 1, 0)
            )
            
            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 84
            rel_xyz_emb_input = posenc(
                torch.cat([rel_xyz_forward, rel_xyz_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 126
            rel_xyz_emb_input = self.xyz_mlp(rel_xyz_emb_input)
            transformer_input.append(rel_pos_emb_input)
            transformer_input_depth.append(rel_xyz_emb_input)
            # get the queries world 
            with torch.no_grad():
                # update the query points with scale and shift
                queries_xyz_i = queries_xyz.clone().detach()
                queries_xyz_i[..., -1] = queries_xyz_i[..., -1] * scale_est_query.view(B,1,N) + shift_est_query.view(B,1,N,3)[...,2]
                _, _, q_xyz_cam = self.track_from_cam(queries_xyz_i, self.c2w_est_curr,
                                                intrs, rgbs=None, visualize=False)
                q_xyz_cam = self.norm_xyz(q_xyz_cam)

            query_t = queries[:,None,:,:1].repeat(B, T, 1, 1)
            q_xyz_cam = torch.cat([query_t/T, q_xyz_cam], dim=-1)
            T_all = torch.arange(T, device=device)[None,:,None,None].repeat(B, 1, N, 1)
            current_xyzt = torch.cat([T_all/T, coords_xyz_init.view(B, T, N, -1)], dim=-1)
            rel_pos_query_glob = q_xyz_cam - current_xyzt
            # embed the confidence and dynamic probability
            confidence_curr = torch.sigmoid(confidence[...,None])
            dynamic_prob_curr = torch.sigmoid(dynamic_prob[...,None]).mean(dim=1, keepdim=True).repeat(1,T,1,1)
            # embed the confidence and dynamic probability
            rel_pos_query_glob = torch.cat([rel_pos_query_glob, confidence_curr, dynamic_prob_curr], dim=-1)
            rel_pos_query_glob = posenc(rel_pos_query_glob, min_deg=0, max_deg=12)
            transformer_input_depth.append(rel_pos_query_glob)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )
            x_depth = (
                torch.cat(transformer_input_depth, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )
            x_depth = self.proj_xyz_embed(x_depth)

            x = x + self.interpolate_time_embed(x, T)
            x = x.view(B, N, T, -1)  # (B N) T D -> B N T D
            x_depth = x_depth + self.interpolate_time_embed(x_depth, T)
            x_depth = x_depth.view(B, N, T, -1)  # (B N) T D -> B N T D
            delta, delta_depth, delta_dynamic_prob, delta_pro_analysis_w, scale_shift_out, dense_res_out = self.updateformer3D(
                x,
                x_depth, 
                self.updateformer,
                add_space_attn=add_space_attn,
                extra_sparse_tokens=extra_sparse_tokens,
                extra_dense_tokens=extra_dense_tokens,
            )
            # update the scale and shift
            if scale_shift_out is not None:
                extra_sparse_tokens = extra_sparse_tokens + scale_shift_out[...,:128]
                scale_update = scale_shift_out[:,:1,:,-1].permute(0,2,1)[...,None]
                shift_update = scale_shift_out[:,1:,:,-1].permute(0,2,1)[...,None]
                scale_est = scale_est + scale_update
                shift_est[...,2:] = shift_est[...,2:] + shift_update / 10
                # dense tokens update
                extra_dense_tokens = extra_dense_tokens + dense_res_out[:,:,-128:]
                res_low = dense_res_out[:,:,:3]
                up_mask = self.upsample_transformer(extra_dense_tokens.mean(dim=1), res_low)
                up_mask = repeat(up_mask, "b k h w -> b s k h w", s=T)
                up_mask = rearrange(up_mask, "b s c h w -> (b s) 1 c h w")
                res_up = self.upsample_with_mask(
                        rearrange(res_low, 'b t c h w -> (b t) c h w'),
                        up_mask,
                    )
                res_up = rearrange(res_up, "(b t) c h w -> b t c h w", b=B, t=T)
                # residual_point = residual_point + res_up
            
            delta_coords = delta[..., :D_coords].permute(0, 2, 1, 3)
            delta_vis = delta[..., D_coords].permute(0, 2, 1)
            delta_confidence = delta[..., D_coords + 1].permute(0, 2, 1)

            vis = vis + delta_vis
            confidence = confidence + delta_confidence
            dynamic_prob = dynamic_prob + delta_dynamic_prob[...,0].permute(0, 2, 1)
            pro_analysis_w = pro_analysis_w + delta_pro_analysis_w[...,0].permute(0, 2, 1)
            # update the depth
            vis_est = torch.sigmoid(vis.detach())

            delta_xyz = delta_depth[...,:3].permute(0,2,1,3)
            denorm_delta_depth = (self.denorm_xyz(coords_xyz+delta_xyz)-self.denorm_xyz(coords_xyz))[...,2:3]


            delta_depth_ = denorm_delta_depth.detach()
            delta_coords = torch.cat([delta_coords, delta_depth_],dim=-1)
            coords = coords + delta_coords
            coords_append = coords.clone()
            coords_xyz_append = self.denorm_xyz(coords_xyz + delta_xyz).clone()

            coords_append[..., :2] = coords_append[..., :2] * float(self.stride) 
            coords_append[..., 0] /= self.factor_x
            coords_append[..., 1] /= self.factor_y

            # get the camera pose from tracks
            dynamic_prob_curr = torch.sigmoid(dynamic_prob.detach())*torch.sigmoid(pro_analysis_w)
            mask_out = (coords_append[...,0]<W_)&(coords_append[...,0]>0)&(coords_append[...,1]<H_)&(coords_append[...,1]>0)
            if query_no_BA:
                dynamic_prob_curr[:,:,:ba_len] = torch.ones_like(dynamic_prob_curr[:,:,:ba_len])
            point_map_org_i = scale_est.view(B*T,1,1,1)*point_map_org.clone().detach() + shift_est.view(B*T,3,1,1)
            # depth_unproj = bilinear_sampler(point_map_org_i, coords_append[...,:2].view(B*T, N, 1, 2), mode="nearest")[:,2,:,0].detach()   
            
            depth_unproj_neg = self.get_correlation_feat(
                    point_map_org_i.view(B,T,3,point_map_org_i.shape[-2], point_map_org_i.shape[-1]),
                     coords_append[...,:2].view(B*T, N, 2), radius=self.corr3d_radius
                )[..., 2]
            depth_diff = (depth_unproj_neg.view(B,T,N,-1) - coords_append[...,2:]).abs()
            idx_neg = torch.argmin(depth_diff, dim=-1)
            depth_unproj = depth_unproj_neg.view(B,T,N,-1)[torch.arange(B)[:, None, None, None],
                                                          torch.arange(T)[None, :, None, None],
                                                          torch.arange(N)[None, None, :, None],
                                                          idx_neg.view(B,T,N,1)].view(B*T, N)
            
            unc_unproj = bilinear_sampler(self.metric_unc_org, coords_append[...,:2].view(B*T, N, 1, 2), mode="nearest")[:,0,:,0].detach()
            depth_unproj[unc_unproj<0.5] = 0.0

            # replace the depth for visible and solid points
            conf_est = torch.sigmoid(confidence.detach())
            replace_mask = (depth_unproj.view(B,T,N)>0.0) * (vis_est>0.5) # * (conf_est>0.5)
            #NOTE: way1: find the jitter points
            depth_rel = (depth_unproj.view(B, T, N) - queries_z.permute(0, 2, 1))
            depth_ddt1 = depth_rel[:, 1:, :] - depth_rel[:, :-1, :]
            depth_ddt2 = depth_rel[:, 2:, :] - 2 * depth_rel[:, 1:-1, :] + depth_rel[:, :-2, :]
            jitter_mask = torch.zeros_like(depth_rel, dtype=torch.bool)
            if depth_ddt2.abs().max()>0:
                thre2 = torch.quantile(depth_ddt2.abs()[depth_ddt2.abs()>0], replace_ratio)
                jitter_mask[:, 1:-1, :] = (depth_ddt2.abs() < thre2)  
                thre1 = torch.quantile(depth_ddt1.abs()[depth_ddt1.abs()>0], replace_ratio)
                jitter_mask[:, :-1, :] *= (depth_ddt1.abs() < thre1)
                replace_mask = replace_mask * jitter_mask

            #NOTE: way2: top k topological change detection
            # coords_2d_lift = coords_append.clone()
            # coords_2d_lift[...,2][replace_mask] = depth_unproj.view(B,T,N)[replace_mask]
            # coords_2d_lift = self.cam_from_track(coords_2d_lift.clone(), intrs_org, only_cam_pts=True)
            # coords_2d_lift[~replace_mask] = coords_xyz_append[~replace_mask]
            # import pdb; pdb.set_trace()
            # jitter_mask = get_topo_mask(coords_xyz_append, coords_2d_lift, replace_ratio)
            # replace_mask = replace_mask * jitter_mask
            
            # replace the depth
            if self.training:
                replace_mask = torch.zeros_like(replace_mask)
            coords_append[...,2][replace_mask] = depth_unproj.view(B,T,N)[replace_mask]
            coords_xyz_unproj = self.cam_from_track(coords_append.clone(), intrs_org, only_cam_pts=True)
            coords[...,2][replace_mask] = depth_unproj.view(B,T,N)[replace_mask]
            # coords_xyz_append[replace_mask] = coords_xyz_unproj[replace_mask]
            coords_xyz_append_refine = coords_xyz_append.clone()
            coords_xyz_append_refine[replace_mask] = coords_xyz_unproj[replace_mask]

            c2w_traj_est, cam_pts_est, intrs_refine, coords_refine, world_tracks, world_tracks_refined, c2w_traj_init = self.cam_from_track(coords_append.clone(),
                                                  intrs_org, dynamic_prob_curr, queries_z_unc, conf_est*vis_est*mask_out.float(),
                                                  track_feat_concat=x_depth, tracks_xyz=coords_xyz_append_refine, init_pose=init_pose,
                                                  query_pts=queries_xyz_i, fixed_cam=fixed_cam, depth_unproj=depth_unproj, cam_gt=cam_gt)
            intrs_org = intrs_refine.view(B, T, 3, 3).to(intrs_org.dtype)
            
            # get the queries world 
            self.c2w_est_curr = c2w_traj_est.detach()
            
            # update coords and coords_append
            coords[..., 2] = (cam_pts_est)[...,2]
            coords_append[..., 2] = (cam_pts_est)[...,2]

            # update coords_xyz_append
            # coords_xyz_append = cam_pts_est
            coords_xyz = self.norm_xyz(cam_pts_est)

           
            # proj 
            coords_xyz_de = coords_xyz_append.clone()
            coords_xyz_de[coords_xyz_de[...,2].abs()<1e-6] = -1e-4
            mask_nan = coords_xyz_de[...,2].abs()<1e-2
            coords_proj = torch.einsum("btij,btnj->btni", intrs_org, coords_xyz_de/coords_xyz_de[...,2:3].abs())[...,:2]
            coords_proj[...,0] *= self.factor_x
            coords_proj[...,1] *= self.factor_y
            coords_proj[...,:2] /= float(self.stride)
            # make sure it is aligned with 2d tracking
            coords_proj_curr = coords[...,:2].view(B*T, N, 2).detach()
            vis_est = (vis_est>0.5).float()
            sync_loss += (vis_est.detach()[...,None]*(coords_proj_curr - coords_proj).norm(dim=-1, keepdim=True)*(1-mask_nan[...,None].float())).mean()
            # coords_proj_curr[~mask_nan.view(B*T, N)] = coords_proj.view(B*T, N, 2)[~mask_nan.view(B*T, N)].to(coords_proj_curr.dtype)

            #NOTE: the 2d tracking + unproject depth
            fix_cam_est = coords_append.clone()
            fix_cam_est[...,2] = depth_unproj
            fix_cam_pts = self.cam_from_track(
                        fix_cam_est, intrs_org, only_cam_pts=True
                    )
            
            coord_preds.append(coords_append)
            coords_xyz_preds.append(coords_xyz_append)
            vis_preds.append(vis)
            cam_preds.append(c2w_traj_init)
            pts3d_cam_pred.append(cam_pts_est)
            world_tracks_pred.append(world_tracks)
            world_tracks_refined_pred.append(world_tracks_refined)
            confidence_preds.append(confidence)
            dynamic_prob_preds.append(dynamic_prob)
            scale_ests.append(scale_est)
            shift_ests.append(shift_est)

        if stage!=0:
            all_coords_predictions.append([coord for coord in coord_preds])
            all_coords_xyz_predictions.append([coord_xyz for coord_xyz in coords_xyz_preds])
            all_vis_predictions.append(vis_preds)
            all_confidence_predictions.append(confidence_preds)
            all_dynamic_prob_predictions.append(dynamic_prob_preds)
            all_cam_predictions.append([cam for cam in cam_preds])
            all_cam_pts_predictions.append([pts for pts in pts3d_cam_pred])
            all_world_tracks_predictions.append([world_tracks for world_tracks in world_tracks_pred])
            all_world_tracks_refined_predictions.append([world_tracks_refined for world_tracks_refined in world_tracks_refined_pred])
            all_scale_est.append(scale_ests)
            all_shift_est.append(shift_ests)
        if stage!=0:
            train_data = (
                all_coords_predictions,
                all_coords_xyz_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                all_dynamic_prob_predictions,
                all_cam_predictions,
                all_cam_pts_predictions,
                all_world_tracks_predictions,
                all_world_tracks_refined_predictions,
                all_scale_est,
                all_shift_est,
                torch.ones_like(vis_preds[-1], device=vis_preds[-1].device),
            )
        else:
            train_data = None
        # resize back 
        # init the trajectories by camera motion
        
        # if cache is not None:
        #     viser = Visualizer(save_dir=".", grayscale=True, 
        #                        fps=10, pad_value=50, tracks_leave_trace=0)
        #     coords_clone = coords.clone()
        #     coords_clone[...,:2] *= self.stride
        #     coords_clone[..., 0] /= self.factor_x
        #     coords_clone[..., 1] /= self.factor_y
        #     viser.visualize(video=video_vis, tracks=coords_clone[..., :2], filename="test_refine")
        #     import pdb; pdb.set_trace()

        if train_data is not None:
            # get the gt pts in the world coordinate
            self_supervised = False
            if (traj3d_gt is not None):
                if traj3d_gt[...,2].abs().max()>0:
                    gt_cam_pts = self.cam_from_track(
                        traj3d_gt, intrs_org, only_cam_pts=True
                    )
                else:
                    self_supervised = True
            else:
                self_supervised = True
            
            if self_supervised:
                gt_cam_pts = self.cam_from_track(
                    coord_preds[-1].detach(), intrs_org, only_cam_pts=True
                )

            if cam_gt is not None:
                gt_world_pts = torch.einsum(
                    "btij,btnj->btni", 
                    cam_gt[...,:3,:3],
                    gt_cam_pts
                ) + cam_gt[...,None, :3,3]  # B T N 3
            else:
                gt_world_pts = torch.einsum(
                    "btij,btnj->btni", 
                    self.c2w_est_curr[...,:3,:3],
                    gt_cam_pts
                ) + self.c2w_est_curr[...,None, :3,3]  # B T N 3
                # update the query points with scale and shift
                queries_xyz_i = queries_xyz.clone().detach()
                queries_xyz_i[..., -1] = queries_xyz_i[..., -1] * scale_est_query.view(B,1,N) + shift_est_query.view(B,1,N,3)[...,2]    
                q_static_proj, q_xyz_world, q_xyz_cam = self.track_from_cam(queries_xyz_i,
                     self.c2w_est_curr,
                    intrs, rgbs=video_vis, visualize=False)
            
                q_static_proj[..., 0] /= self.factor_x
                q_static_proj[..., 1] /= self.factor_y
                cam_gt = self.c2w_est_curr[:,:,:3,:]

            if traj3d_gt is not None:
                ret_loss = self.loss(train_data, traj3d_gt,
                                      vis_gt, None, cam_gt, queries_z_unc,
                                      q_xyz_world, q_static_proj, anchor_loss=anchor_loss, fix_cam_pts=fix_cam_pts, video_vis=video_vis, stage=stage,
                                      gt_world_pts=gt_world_pts, mask_traj_gt=mask_traj_gt, intrs=intrs_org, custom_vid=custom_vid, valid_only=valid_only,
                                      c2w_ests=c2w_ests, point_map_preds=point_map_preds, points_map_gt=points_map_gt, metric_unc=metric_unc, scale_est=scale_est,
                                      shift_est=shift_est, point_map_org_train=point_map_org_train)
            else:
                ret_loss = self.loss(train_data, traj3d_gt,
                                      vis_gt, None, cam_gt, queries_z_unc,
                                      q_xyz_world, q_static_proj, anchor_loss=anchor_loss, fix_cam_pts=fix_cam_pts, video_vis=video_vis, stage=stage,
                                      gt_world_pts=gt_world_pts, mask_traj_gt=mask_traj_gt, intrs=intrs_org, custom_vid=custom_vid, valid_only=valid_only,
                                      c2w_ests=c2w_ests, point_map_preds=point_map_preds, points_map_gt=points_map_gt, metric_unc=metric_unc, scale_est=scale_est,
                                      shift_est=shift_est, point_map_org_train=point_map_org_train)
            if custom_vid:
                sync_loss = 0*sync_loss
            if (sync_loss > 50) and (stage==1):
                ret_loss = (0*sync_loss, 0*sync_loss, 0*sync_loss, 0*sync_loss, 0*sync_loss, 0*sync_loss, 0*sync_loss) + (0*sync_loss,)
            else:
                ret_loss = ret_loss+(10*sync_loss,)

        else:
            ret_loss = None

        color_pts = torch.cat([pts3d_cam_pred[-1], queries_rgb[:,None].repeat(1, T, 1, 1)], dim=-1) 
        
        #TODO: For evaluation. We found our model have some bias on invisible points after training. (to be fixed) 
        vis_pred_out = torch.sigmoid(vis_preds[-1]) + 0.2
        
        ret = {"preds": coord_preds[-1], "vis_pred": vis_pred_out,
                 "conf_pred": torch.sigmoid(confidence_preds[-1]),
                "cam_pred": self.c2w_est_curr,"loss": ret_loss}

        cache = {
            "fmaps": fmaps_org[0].detach(),
            "track_feat_support3d_pyramid": [track_feat_support3d_pyramid[i].detach() for i in range(len(track_feat_support3d_pyramid))],
            "track_point_map_support_pyramid": [self.denorm_xyz(track_point_map_support_pyramid[i].detach()) for i in range(len(track_point_map_support_pyramid))],
            "track_feat3d_pyramid": [track_feat3d_pyramid[i].detach() for i in range(len(track_feat3d_pyramid))],
            "track_point_map_pyramid": [self.denorm_xyz(track_point_map_pyramid[i].detach()) for i in range(len(track_point_map_pyramid))],
            "track_feat_pyramid": [track_feat_pyramid[i].detach() for i in range(len(track_feat_pyramid))],
            "track_feat_support_pyramid": [track_feat_support_pyramid[i].detach() for i in range(len(track_feat_support_pyramid))],
            "track2d_pred_cache": coord_preds[-1][0].clone().detach(),
            "track3d_pred_cache": pts3d_cam_pred[-1][0].clone().detach(),
        }
        #NOTE: update the point map
        point_map_org = scale_est.view(B*T,1,1,1)*point_map_org + shift_est.view(B*T,3,1,1)
        point_map_org_refined = point_map_org
        return ret, torch.sigmoid(dynamic_prob_preds[-1])*queries_z_unc[:,None,:,0], coord_preds[-1], color_pts, intrs_org, point_map_org_refined, cache
    
    def track_d2_loss(self, tracks3d, stride=[1,2,3], dyn_prob=None, mask=None):
        """
        tracks3d: B T N 3
        dyn_prob: B T N 1
        """
        r = 0.8
        t_diff_total = 0.0
        for i, s_ in enumerate(stride):
            w_ = r**i
            tracks3d_stride = tracks3d[:, ::s_, :, :]  # B T//s_ N 3
            t_diff_tracks3d = (tracks3d_stride[:, 1:, :, :] - tracks3d_stride[:, :-1, :, :])
            t_diff2 = (t_diff_tracks3d[:, 1:, :, :] - t_diff_tracks3d[:, :-1, :, :])
            t_diff_total += w_*(t_diff2.norm(dim=-1).mean())

        return 1e2*t_diff_total

    def loss(self, train_data, traj3d_gt=None,
                         vis_gt=None, static_tracks_gt=None, cam_gt=None, 
                         z_unc=None, q_xyz_world=None, q_static_proj=None, anchor_loss=0, valid_only=False,
                         gt_world_pts=None, mask_traj_gt=None, intrs=None, c2w_ests=None, custom_vid=False, video_vis=None, stage=0,
                         fix_cam_pts=None, point_map_preds=None, points_map_gt=None, metric_unc=None, scale_est=None, shift_est=None, point_map_org_train=None):
        """
        Compute the loss of 3D tracking problem
        
        """
        
        (
            coord_predictions, coords_xyz_predictions, vis_predictions, confidence_predicitons,
            dynamic_prob_predictions, camera_predictions, cam_pts_predictions, world_tracks_predictions,
            world_tracks_refined_predictions, scale_ests, shift_ests, valid_mask
        ) = train_data
        B, T, _, _ = cam_gt.shape
        if (stage == 2) and self.training:
            # get the scale and shift gt
            self.metric_unc_org[:,0] = self.metric_unc_org[:,0] * (points_map_gt.norm(dim=-1)>0).float() * (self.metric_unc_org[:,0]>0.5).float()
            if not (self.scale_gt==torch.ones(B*T).to(self.scale_gt.device)).all():
                scale_gt, shift_gt = self.scale_gt, self.shift_gt
                scale_re = scale_gt[:4].mean()
                scale_loss = 0.0
                shift_loss = 0.0
                for i_scale in range(len(scale_ests[0])):
                    scale_loss += 0.8**(len(scale_ests[0])-i_scale-1)*10*(scale_gt - scale_re*scale_ests[0][i_scale].view(-1)).abs().mean()
                    shift_loss += 0.8**(len(shift_ests[0])-i_scale-1)*10*(shift_gt - scale_re*shift_ests[0][i_scale].view(-1,3)).abs().mean()
            else:
                scale_loss = 0.0 * scale_ests[0][0].mean()
                shift_loss = 0.0 * shift_ests[0][0].mean()
                scale_re = 1.0
        else:
            scale_loss = 0.0
            shift_loss = 0.0

        if len(point_map_preds)>0:
            point_map_loss = 0.0
            for i in range(len(point_map_preds)):
                point_map_preds_i = point_map_preds[i]
                point_map_preds_i = rearrange(point_map_preds_i, 'b t c h w -> (b t) c h w', b=B, t=T)
                base_loss = ((self.pred_points - points_map_gt).norm(dim=-1) * self.metric_unc_org[:,0]).mean()
                point_map_loss_i = ((point_map_preds_i - points_map_gt.permute(0,3,1,2)).norm(dim=1) * self.metric_unc_org[:,0]).mean()
                point_map_loss += point_map_loss_i
                # point_map_loss += ((point_map_org_train - points_map_gt.permute(0,3,1,2)).norm(dim=1) * self.metric_unc_org[:,0]).mean()
            if scale_loss == 0.0:
                point_map_loss = 0*point_map_preds_i.sum()
        else:
            point_map_loss = 0.0
        
        # camera loss
        cam_loss = 0.0
        dyn_loss = 0.0
        N_gt = gt_world_pts.shape[2]
        
        # self supervised dynamic mask
        H_org, W_org = self.image_size[0], self.image_size[1]
        q_static_proj[torch.isnan(q_static_proj)] = -200
        in_view_mask = (q_static_proj[...,0]>0) & (q_static_proj[...,0]<W_org) & (q_static_proj[...,1]>0) & (q_static_proj[...,1]<H_org)
        dyn_mask_final = (((coord_predictions[0][-1] - q_static_proj))[...,:2].norm(dim=-1) * in_view_mask)
        dyn_mask_final = dyn_mask_final.sum(dim=1) / (in_view_mask.sum(dim=1) + 1e-2)
        dyn_mask_final = dyn_mask_final > 6
        
        for iter_, cam_pred_i in enumerate(camera_predictions[0]):
            # points loss
            pts_i_world = world_tracks_predictions[0][iter_].view(B, T, -1, 3)
            
            coords_xyz_i_world = coords_xyz_predictions[0][iter_].view(B, T, -1, 3)
            coords_i = coord_predictions[0][iter_].view(B, T, -1, 3)[..., :2]
            pts_i_world_refined = torch.einsum(
                "btij,btnj->btni", 
                cam_gt[...,:3,:3],
                coords_xyz_i_world
            ) + cam_gt[...,None, :3,3]  # B T N 3

            # pts_i_world_refined = world_tracks_refined_predictions[0][iter_].view(B, T, -1, 3)
            pts_world = pts_i_world
            dyn_prob_i_logits = dynamic_prob_predictions[0][iter_].mean(dim=1)
            dyn_prob_i = torch.sigmoid(dyn_prob_i_logits).detach()
            mask = pts_world.norm(dim=-1) < 200
            
            # general 
            vis_i_logits = vis_predictions[0][iter_]
            vis_i = torch.sigmoid(vis_i_logits).detach()
            if mask_traj_gt is not None:    
                try:
                    N_gt_mask = mask_traj_gt.shape[1]
                    align_loss = (gt_world_pts - q_xyz_world[:,None,:N_gt,:,]).norm(dim=-1)[...,:N_gt_mask] * (mask_traj_gt.permute(0,2,1))
                    visb_traj = (align_loss * vis_i[:,:,:N_gt_mask]).sum(dim=1)/vis_i[:,:,:N_gt_mask].sum(dim=1)
                except:
                    import pdb; pdb.set_trace()
            else:
                visb_traj = ((gt_world_pts - q_xyz_world[:,None,:N_gt,:,]).norm(dim=-1) * vis_i[:,:,:N_gt]).sum(dim=1)/vis_i[:,:,:N_gt].sum(dim=1)

            # pts_loss = ((q_xyz_world[:,None,...] - pts_world)[:,:,:N_gt,:].norm(dim=-1)*(1-dyn_prob_i[:,None,:N_gt])) # - 0.1*(1-dyn_prob_i[:,None,:N_gt]).log()
            pts_loss = 0
            static_mask = ~dyn_mask_final   # more strict for static points
            dyn_mask = dyn_mask_final
            pts_loss_refined = ((q_xyz_world[:,None,...] - pts_i_world_refined).norm(dim=-1)*static_mask[:,None,:]).sum()/static_mask.sum() # - 0.1*(1-dyn_prob_i[:,None,:N_gt]).log()
            vis_logits_final = vis_predictions[0][-1].detach()
            vis_final = torch.sigmoid(vis_logits_final)+0.2 > 0.5  # more strict for visible points
            dyn_vis_mask = dyn_mask*vis_final * (fix_cam_pts[...,2] > 0.1)
            pts_loss_dynamic = ((fix_cam_pts - coords_xyz_i_world).norm(dim=-1)*dyn_vis_mask[:,None,:]).sum()/dyn_vis_mask.sum()
            
            # pts_loss_refined = 0
            if traj3d_gt is not None:
                tap_traj = (gt_world_pts[:,:-1,...] - gt_world_pts[:,1:,...]).norm(dim=-1).sum(dim=1)[...,:N_gt_mask]
                mask_dyn = tap_traj>0.5
                if mask_traj_gt.sum() > 0:
                    dyn_loss_i = 20*balanced_binary_cross_entropy(dyn_prob_i_logits[:,:N_gt_mask][mask_traj_gt.squeeze(-1)], 
                                                                                        mask_dyn.float()[mask_traj_gt.squeeze(-1)])
                else:
                    dyn_loss_i = 0
            else:                
                dyn_loss_i = 10*balanced_binary_cross_entropy(dyn_prob_i_logits, dyn_mask_final.float())
            
            dyn_loss += dyn_loss_i

            # visible loss for out of view points
            vis_i_train = torch.sigmoid(vis_i_logits)
            out_of_view_mask = (coords_i[...,0]<0)|(coords_i[...,0]>self.image_size[1])|(coords_i[...,1]<0)|(coords_i[...,1]>self.image_size[0])
            vis_loss_out_of_view = vis_i_train[out_of_view_mask].sum() / out_of_view_mask.sum()


            if traj3d_gt is not None:
                world_pts_loss = (((gt_world_pts - pts_i_world_refined[:,:,:gt_world_pts.shape[2],...]).norm(dim=-1))[...,:N_gt_mask] * mask_traj_gt.permute(0,2,1)).sum() / mask_traj_gt.sum()
                # world_pts_init_loss = (((gt_world_pts - pts_i_world[:,:,:gt_world_pts.shape[2],...]).norm(dim=-1))[...,:N_gt_mask] * mask_traj_gt.permute(0,2,1)).sum() / mask_traj_gt.sum()
            else:
                world_pts_loss = 0
            
            # cam regress
            t_err = (cam_pred_i[...,:3,3] - cam_gt[...,:3,3]).norm(dim=-1).sum()
            
            # xyz loss
            in_view_mask_large = (q_static_proj[...,0]>-50) & (q_static_proj[...,0]<W_org+50) & (q_static_proj[...,1]>-50) & (q_static_proj[...,1]<H_org+50)
            static_vis_mask = (q_static_proj[...,2]>0.05).float() * static_mask[:,None,:] * in_view_mask_large
            xyz_loss = ((coord_predictions[0][iter_] - q_static_proj)).abs()[...,:2].norm(dim=-1)*static_vis_mask
            xyz_loss = xyz_loss.sum()/static_vis_mask.sum()
            
            # visualize the q_static_proj
            # viser = Visualizer(save_dir=".", grayscale=True, 
            #                     fps=10, pad_value=50, tracks_leave_trace=0)
            # video_vis_ = F.interpolate(video_vis.view(B*T,3,video_vis.shape[-2],video_vis.shape[-1]), (H_org, W_org), mode='bilinear', align_corners=False)
            # viser.visualize(video=video_vis_, tracks=q_static_proj[:,:,dyn_mask_final.squeeze(), :2], filename="test")
            # viser.visualize(video=video_vis_, tracks=coord_predictions[0][-1][:,:,dyn_mask_final.squeeze(), :2], filename="test_pred")
            # import pdb; pdb.set_trace()

            # temporal loss
            t_loss = self.track_d2_loss(pts_i_world_refined, [1,2,3], dyn_prob=dyn_prob_i, mask=mask)
            R_err = (cam_pred_i[...,:3,:3] - cam_gt[...,:3,:3]).abs().sum(dim=-1).mean()
            if self.stage == 1:
                cam_loss += 0.8**(len(camera_predictions[0])-iter_-1)*(10*t_err + 500*R_err + 20*pts_loss_refined + 10*xyz_loss + 20*pts_loss_dynamic + 10*vis_loss_out_of_view) #+ 5*(pts_loss + pts_loss_refined + world_pts_loss) + t_loss)
            elif self.stage == 3:
                cam_loss += 0.8**(len(camera_predictions[0])-iter_-1)*(10*t_err + 500*R_err + 10*vis_loss_out_of_view) #+ 5*(pts_loss + pts_loss_refined + world_pts_loss) + t_loss)
            else:
                cam_loss += 0*vis_loss_out_of_view

        if (cam_loss > 20000)|(torch.isnan(cam_loss)):
            cam_loss = torch.zeros_like(cam_loss)


        if traj3d_gt is None:
        # ================ Condition 1: The self-supervised signals from the self-consistency ===================
            return cam_loss, train_data[0][0][0].mean()*0, dyn_loss, train_data[0][0][0].mean()*0, point_map_loss, scale_loss, shift_loss


        # ================ Condition 2: The supervision signal given by the ground truth trajectories ===================
        if (
            (torch.isnan(traj3d_gt).any()
            or traj3d_gt.abs().max() > 2000) and (custom_vid==False)
        ):
            return cam_loss, train_data[0][0][0].mean()*0, dyn_loss, train_data[0][0][0].mean()*0, point_map_loss, scale_loss, shift_loss


        vis_gts = [vis_gt.float()]
        invis_gts = [1-vis_gt.float()]
        traj_gts = [traj3d_gt]
        valids_gts = [valid_mask]
        seq_loss_all = sequence_loss(
            coord_predictions,
            traj_gts,
            valids_gts,
            vis=vis_gts,
            gamma=0.8,
            add_huber_loss=False,
            loss_only_for_visible=False if custom_vid==False else True,
            z_unc=z_unc,
            mask_traj_gt=mask_traj_gt
        )

        confidence_loss = sequence_prob_loss(
            coord_predictions, confidence_predicitons, traj_gts, vis_gts
        )

        seq_loss_xyz = sequence_loss_xyz(
            coords_xyz_predictions,
            traj_gts,
            valids_gts,
            intrs=intrs,
            vis=vis_gts,
            gamma=0.8,
            add_huber_loss=False,
            loss_only_for_visible=False,
            mask_traj_gt=mask_traj_gt
        )

        # filter the blinking points
        mask_vis = vis_gts[0].clone()  # B T N
        mask_vis[mask_vis==0] = -1
        blink_mask = mask_vis[:,:-1,:] * mask_vis[:,1:,:] # first derivative   B (T-1) N
        mask_vis[:,:-1,:], mask_vis[:,-1,:] = (blink_mask == 1), 0

        vis_loss = sequence_BCE_loss(vis_predictions, vis_gts, mask=[mask_vis])

        track_loss_out = (seq_loss_all+2*seq_loss_xyz + cam_loss)
        if valid_only:
            vis_loss = 0.0*vis_loss
        if custom_vid:
            return seq_loss_all, 0.0*seq_loss_all, 0.0*seq_loss_all, 10*vis_loss, 0.0*seq_loss_all, 0.0*seq_loss_all, 0.0*seq_loss_all

        return track_loss_out, confidence_loss, dyn_loss, 10*vis_loss, point_map_loss, scale_loss, shift_loss




