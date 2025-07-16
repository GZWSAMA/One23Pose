# 🌟 SpatialTrackerV2 Integrated with SAM 🌟
SAM receives a point prompt and generates a mask for the target object, facilitating easy interaction to obtain the object's 3D trajectories with SpaTrack2.

## Installation
```

python -m pip install git+https://github.com/facebookresearch/segment-anything.git
cd app_3rd/sam_utils
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```