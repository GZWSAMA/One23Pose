# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from tqdm import tqdm
from models.SpaTrackV2.models.SpaTrack import SpaTrack2
from typing import Literal
import numpy as np
from pathlib import Path
from typing import Union, Optional
import cv2
import os
# import decord
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

def load_video_with_opencv(path, fps=25):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {path}")

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 按照指定帧率抽帧（例如每第 fps 帧保留一次）
        if frame_count % fps == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            frames.append(frame)
        frame_count += 1

    cap.release()

    # 转为 numpy array 并转换成 tensor
    video_array = np.array(frames)
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)

    return video_tensor

class Predictor(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.spatrack = SpaTrack2(loggers=[None, None, None], **args)
        self.S_wind = args["Track_cfg"]["s_wind"]
        self.overlap = args["Track_cfg"]["overlap"]

    def to(self, device: Union[str, torch.device]):
        self.spatrack.to(device)
        if self.spatrack.base_model is not None:
            self.spatrack.base_model.to(device)

    def forward(self, video: str|torch.Tensor|np.ndarray,
                 depth: str|torch.Tensor|np.ndarray=None,
                 unc_metric: str|torch.Tensor|np.ndarray=None,
                 intrs: str|torch.Tensor|np.ndarray=None,
                 extrs: str|torch.Tensor|np.ndarray=None,
                 queries=None, queries_3d=None, iters_track=4,
                 full_point=False, fps=30, track2d_gt=None, 
                 fixed_cam=False, query_no_BA=False, stage=0,
                 support_frame=0, replace_ratio=0.6):
        """
        video: this could be a path to a video, a tensor of shape (T, C, H, W) or a numpy array of shape (T, C, H, W)
        queries: (B, N, 2)
        """

        if isinstance(video, str):
            # video = decord.VideoReader(video)
            # video = video[::fps].asnumpy()  # Convert to numpy array
            # video = np.array(video)  # Ensure numpy array
            # video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
            video = load_video_with_opencv(video, fps)
        elif isinstance(video, np.ndarray):
            video = torch.from_numpy(video).float()

        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        if isinstance(intrs, np.ndarray):
            intrs = torch.from_numpy(intrs).float()
        if isinstance(extrs, np.ndarray):
            extrs = torch.from_numpy(extrs).float()
        if isinstance(unc_metric, np.ndarray):
            unc_metric = torch.from_numpy(unc_metric).float()

        T_, C, H, W = video.shape
        step_slide = self.S_wind - self.overlap
        if T_ > self.S_wind:
            
            num_windows = (T_ - self.S_wind + step_slide) // step_slide 
            T = num_windows * step_slide + self.S_wind
            pad_len = T - T_

            video = torch.cat([video, video[-1:].repeat(T-video.shape[0], 1, 1, 1)], dim=0)
            if depth is not None:
                depth = torch.cat([depth, depth[-1:].repeat(T-depth.shape[0], 1, 1)], dim=0)
            if intrs is not None:
                intrs = torch.cat([intrs, intrs[-1:].repeat(T-intrs.shape[0], 1, 1)], dim=0)
            if extrs is not None:
                extrs = torch.cat([extrs, extrs[-1:].repeat(T-extrs.shape[0], 1, 1)], dim=0)
            if unc_metric is not None:
                unc_metric = torch.cat([unc_metric, unc_metric[-1:].repeat(T-unc_metric.shape[0], 1, 1)], dim=0)
        with torch.no_grad():
            ret = self.spatrack.forward_stream(video, queries, T_org=T_,
                                                depth=depth, intrs=intrs, unc_metric_in=unc_metric, extrs=extrs, queries_3d=queries_3d,
                                                window_len=self.S_wind, overlap_len=self.overlap, track2d_gt=track2d_gt, full_point=full_point, iters_track=iters_track,
                                                fixed_cam=fixed_cam, query_no_BA=query_no_BA, stage=stage, support_frame=support_frame, replace_ratio=replace_ratio) + (video[:T_],)
            
        return ret

    
        
        