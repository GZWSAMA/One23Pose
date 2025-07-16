from models.SpaTrackV2.models.predictor import Predictor
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from rich import print
from torchvision import transforms as TF
import decord
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

def load_and_preprocess_numpy_img(depth_path, mode="crop", test_mode=False):
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    # Open image
    img = Image.open(depth_path)

    if img.mode == "RGBA":
        # Create white background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        # Alpha composite onto the white background
        img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

    width, height = img.size
    
    if mode == "pad":
        # Make the largest dimension 518px while maintaining aspect ratio
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
    else:  # mode == "crop"
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

    # Resize with new dimensions (width, height)
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    if test_mode:
        img.save(f"output/test/middle_images/resized_depth_image.png")
    img = to_tensor(img)  # Convert to tensor (0, 1)

    # Center crop height if it's larger than 518 (only in crop mode)
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]
    
    # For pad mode, pad to make a square of target_size x target_size
    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]
        
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            
            # Pad with white (value=1.0)
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

    shapes.add((img.shape[1], img.shape[2]))
    images.append(img)
    if test_mode:
        img_np = img.detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Step 2: CHW -> HWC 转换
        img_np = np.transpose(img_np, (1, 2, 0))

        # Step 3: 创建 PIL 图像并保存
        pil_img = Image.fromarray(img_np)
        pil_img.save("output/test/middle_images/cropped_depth_image.png")

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    
    images = torch.stack(images)  # concatenate images

    image = images[0].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC，并转为 numpy
    image = (image * 255).clip(0, 255).astype(np.uint8)  # 归一化到 [0, 255] 并转为 uint8

    return image

def process_img_to_vggt_format(image_names, output_path):
    """
    Load and preprocess an image to the format expected by the VGGT model.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load and preprocess the image
    processed_image_names = []
    for i,image_name in enumerate(image_names):
        image = load_and_preprocess_numpy_img(image_name)
        #convert to PIL Image for saving
        image = Image.fromarray(image)
        image.save(f'{output_path}/{i:06d}.jpg')
        processed_image_names.append(f'{output_path}/{i:06d}.jpg')
    return processed_image_names  # Add batch dimension

def initialize_vggt_model(device, vo_points=756):
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to(device)

    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    model.spatrack.track_num = vo_points
    model.eval()
    model.to("cuda")

    return vggt4track_model, model

def estimate_depth_and_intrinsic(vggt4track_model, model, vid_dir, output_path, fps=1, grid_size=10, device='cuda'):
    video_reader = decord.VideoReader(vid_dir)
    video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)  # Convert to tensor and permute to (N, C, H, W)
    video_tensor = video_tensor[::fps].float()

    # process the image tensor
    video_tensor = preprocess_image(video_tensor)[None]
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt4track_model(video_tensor.cuda()/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
    
    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor.squeeze()
    #NOTE: 20% of the depth is not reliable
    # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
    intrinsics = intrs.cpu().numpy()
    depth_save = point_map[:,2,...]
    depth_save[conf_depth<0.5] = 0
    depth_maps = depth_save.cpu().numpy() 

    depth_names = []
    for frame_id, depth_map in enumerate(depth_maps):
        depth_map_mm = (depth_map * 1000).astype('uint16')
        depth_path = f"{output_path}/{frame_id:06d}.png"
        cv2.imwrite(depth_path, depth_map_mm)
        depth_names.append(depth_path)

    return depth_names, intrinsics