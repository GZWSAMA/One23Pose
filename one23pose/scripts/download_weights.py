from huggingface_hub import snapshot_download
import urllib.request
import os

# 下载模型
def download_model_from_hf(repo_id, out_dir):
    print(f"✅ Download {repo_id} to {out_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",  # 可省略，默认为 model
        local_dir=out_dir,  # 本地保存路径
        local_dir_use_symlinks=False,  # 如果你希望实际复制文件而不是使用软链接
        # endpoint="https://hf-mirror.com/"
    )

def download_file_from_google(url, filename):
    """Download a file from a given URL and save it as filename."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Saved to {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to download checkpoint from {url}") from e
    
# 模型仓库ID
repo_id_stable3dgen = "Stable-X/trellis-normal-v0-1"
repo_id_trellis = "jetx/TRELLIS-image-large"
repo_id_spatracker_vggt = "Yuxihenry/SpatialTrackerV2_Front"
repo_id_spatracker_tracker_offline = "Yuxihenry/SpatialTrackerV2-Offline"
repo_id_spatracker_tracker_online = "Yuxihenry/SpatialTrackerV2-Online"

checkpoint_dir = "checkpoints"

# output_dir
hf_model_dict = {
    # repo_id_stable3dgen : f"{checkpoint_dir}/Stable3DGen",
    repo_id_trellis : f"{checkpoint_dir}/Trellis",
    repo_id_spatracker_vggt : f"{checkpoint_dir}/SpatialTrackerV2/vggt_front",
    repo_id_spatracker_tracker_offline : f"{checkpoint_dir}/SpatialTrackerV2/tracker_offline",
    repo_id_spatracker_tracker_online : f"{checkpoint_dir}/SpatialTrackerV2/tracker_online",
}

for key,value in hf_model_dict.items():
    download_model_from_hf(key, value)


# SAM2
# Base URL for the checkpoints
BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"

# Define checkpoint URLs
CHECKPOINTS = {
    "sam2_hiera_tiny.pt": BASE_URL + "sam2_hiera_tiny.pt",
    "sam2_hiera_small.pt": BASE_URL + "sam2_hiera_small.pt",
    "sam2_hiera_base_plus.pt": BASE_URL + "sam2_hiera_base_plus.pt",
    "sam2_hiera_large.pt": BASE_URL + "sam2_hiera_large.pt"
}

for filename, url in CHECKPOINTS.items():
    full_path = os.path.join("checkpoints/SAM2", filename)
    if os.path.exists(full_path):
        print(f"{full_path} already exists. Skipping download.")
        continue
    try:
        download_file_from_google(url, full_path)
    except RuntimeError as e:
        print(e)
        exit(1)

    print("All checkpoints are downloaded successfully.")