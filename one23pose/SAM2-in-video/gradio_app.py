import os
import torch
import numpy as np
import cv2
import gradio as gr
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
import tempfile
import zipfile
import subprocess

# Enable automatic mixed precision for CUDA to improve performance and memory efficiency
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Set environment variable to enable CuDNN backend
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

# Check if the GPU supports TensorFloat-32 (TF32) precision for faster computations
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

predictor = None

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to SAM2-in-video!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

def convert_video_to_mp4(input_path, output_path):
    """Convert video to MP4 format using ffmpeg."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-y',
        output_path
    ]
    subprocess.run(command, check=True)

def initialize_predictor(checkpoint):
    """Initialize the SAM2 video predictor with the specified checkpoint."""
    global predictor
    if checkpoint == "tiny":
        sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
    elif checkpoint == "small":
        sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
    elif checkpoint == "base-plus":
        sam2_checkpoint = "./checkpoints/sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b+.yaml"
    elif checkpoint == "large":
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
    else:
        raise ValueError("Invalid checkpoint")

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def extract_frames(video_path):
    """Extract frames from the input video."""
    if not video_path:
        gr.Warning("No input video")
        return None, None, {}, None, None

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    temp_dir = tempfile.mkdtemp()
    frames = []

    for i in range(frame_count):
        success, frame = video.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        else:
            break

    video.release()

    return frames[0], frames, {}, temp_dir, fps

def add_point_and_segment(image, frames, objects, temp_dir, frame_index, object_id, point_type, evt: gr.SelectData):
    """Add a point to the frame and perform segmentation."""
    if image is None or frames is None or temp_dir is None:
        return image, frames, objects, temp_dir, frame_index

    x, y = evt.index[0], evt.index[1]

    if object_id not in objects:
        objects[object_id] = {"points": [], "mask": None, "color": plt.get_cmap("tab10")(len(objects) % 10)[:3]}

    objects[object_id]["points"].append((x, y, point_type))

    # Prepare points for SAM2
    np_points = np.array([[p[0], p[1]] for p in objects[object_id]["points"]], dtype=np.float32)
    labels = np.array([1 if p[2] == "positive" else 0 for p in objects[object_id]["points"]], dtype=np.int32)

    # Initialize SAM2 state
    inference_state = predictor.init_state(video_path=temp_dir)
    predictor.reset_state(inference_state)

    # Perform segmentation
    _, _, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_index,
        obj_id=object_id,
        points=np_points,
        labels=labels,
    )

    # Update mask
    objects[object_id]["mask"] = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

    # Create a copy of the current frame to draw on
    image_with_points_and_mask = frames[frame_index].copy()

    # Apply masks and draw points for all objects
    for obj_id, obj_data in objects.items():
        color = obj_data["color"]
        mask = obj_data["mask"]
        if mask is not None:
            image_with_points_and_mask[mask] = image_with_points_and_mask[mask] * 0.7 + np.array(color) * 255 * 0.3

        for px, py, pt in obj_data["points"]:
            point_color = (0, 255, 0) if pt == "positive" else (255, 0, 0)
            cv2.circle(image_with_points_and_mask, (px, py), 5, point_color, -1)

    return image_with_points_and_mask, frames, objects, temp_dir, frame_index

def clear_points(frames, objects, temp_dir, frame_index):
    """Clear all points and masks from the current frame."""
    if frames is None:
        return None, None, {}, temp_dir, 0

    # Return the original frame without points or mask
    return frames[frame_index].copy(), frames, {}, temp_dir, frame_index

def change_frame(frames, objects, temp_dir, frame_index):
    """Change to a different frame and update the display."""
    if frames is None:
        return None, frames, objects, temp_dir, frame_index

    image_with_points_and_mask = frames[frame_index].copy()

    # Apply masks and draw points for all objects
    for obj_id, obj_data in objects.items():
        color = obj_data["color"]
        mask = obj_data["mask"]
        if mask is not None:
            image_with_points_and_mask[mask] = image_with_points_and_mask[mask] * 0.7 + np.array(color) * 255 * 0.3

        for px, py, pt in obj_data["points"]:
            point_color = (0, 255, 0) if pt == "positive" else (255, 0, 0)
            cv2.circle(image_with_points_and_mask, (px, py), 5, point_color, -1)

    return image_with_points_and_mask, frames, objects, temp_dir, frame_index

def segment_video(video_path, objects, temp_dir, fps):
    """Segment the entire video based on the annotated points."""
    if not video_path or not objects:
        return None, None  # Return None for both video output and zip file

    frame_names = sorted([p for p in os.listdir(temp_dir) if p.endswith('.jpg')])

    inference_state = predictor.init_state(video_path=temp_dir)
    predictor.reset_state(inference_state)

    # Initial annotation for each object
    for obj_id, obj_data in objects.items():
        np_points = np.array([[p[0], p[1]] for p in obj_data["points"]], dtype=np.float32)
        labels = np.array([1 if p[2] == "positive" else 0 for p in obj_data["points"]], dtype=np.int32)

        predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=np_points,
            labels=labels,
        )

    video_dir = os.path.dirname(video_path)
    output_video_path = os.path.join(video_dir, "output_video.mp4")
    extracted_video_paths = {}

    first_frame = cv2.imread(os.path.join(temp_dir, frame_names[0]))
    height, width = first_frame.shape[:2]

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    object_writers = {}

    for obj_id in objects.keys():
        extracted_video_paths[obj_id] = os.path.join(video_dir, f"extracted_video_obj_{obj_id}.mp4")
        object_writers[obj_id] = cv2.VideoWriter(extracted_video_paths[obj_id], cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame = cv2.imread(os.path.join(temp_dir, frame_names[out_frame_idx]))
        overlay_frame = frame.copy()

        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            color = np.array(objects[out_obj_id]["color"]) * 255

            # For output video with overlay
            overlay_frame[mask] = overlay_frame[mask] * 0.5 + color * 0.5

            # For individual object videos
            object_frame = np.zeros_like(frame)
            object_frame[mask] = frame[mask]
            object_writers[out_obj_id].write(object_frame)

        video_writer.write(overlay_frame)

    video_writer.release()
    for writer in object_writers.values():
        writer.release()

    # Create a zip file containing all extracted videos
    zip_path = os.path.join(video_dir, "extracted_videos.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for obj_id, video_path in extracted_video_paths.items():
            zipf.write(video_path, f"extracted_video_obj_{obj_id}.mp4")

    return output_video_path, zip_path

def sam2_in_video():
    """Create the Gradio interface for SAM2 video segmentation."""
    app = gr.Blocks(js=js,theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]))

    with app:
        gr.Markdown("Video Segmentation with SAM2 - Multiple Object Tracking")
        with gr.Column():
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                image_output = gr.Image(label="Current Frame", interactive=True)

            with gr.Row():
                extract_button = gr.Button("Extract Frames", variant="primary")
                with gr.Row():
                    point_type = gr.Radio(["positive", "negative"], label="Point Type", value="positive")
                    clear_points_button = gr.Button("Clear All Points",variant="stop")

            with gr.Row():
                frame_slider = gr.Slider(label="Frame", minimum=0, maximum=100, step=1, value=0)
                object_id = gr.Number(label="Object ID", value=1, precision=0)

        frames = gr.State(None)
        objects = gr.State({})
        temp_dir = gr.State(None)
        fps = gr.State(None)

        with gr.Row():
            with gr.Column():
                checkpoint = gr.Dropdown(label="Checkpoint", choices=["tiny", "small", "base-plus", "large"], value="tiny")
                segment_button = gr.Button("Segment Video")
                zip_output = gr.DownloadButton(label="Download Extracted Videos (ZIP)")
        
            video_output = gr.Video(label="Segmented Video (All Objects)")

        extract_button.click(
            extract_frames,
            inputs=video_input,
            outputs=[image_output, frames, objects, temp_dir, fps],
            concurrency_limit=1
        )

        image_output.select(
            add_point_and_segment,
            inputs=[image_output, frames, objects, temp_dir, frame_slider, object_id, point_type],
            outputs=[image_output, frames, objects, temp_dir, frame_slider],
            concurrency_limit=1
        )

        clear_points_button.click(
            clear_points,
            inputs=[frames, objects, temp_dir, frame_slider],
            outputs=[image_output, frames, objects, temp_dir, frame_slider],
            concurrency_limit=1
        )

        frame_slider.change(
            change_frame,
            inputs=[frames, objects, temp_dir, frame_slider],
            outputs=[image_output, frames, objects, temp_dir, frame_slider],
            concurrency_limit=1
        )

        segment_button.click(
            segment_video,
            inputs=[video_input, objects, temp_dir, fps],
            outputs=[video_output, zip_output],
            concurrency_limit=1
        )

        checkpoint.change(
            initialize_predictor,
            inputs=checkpoint,
            outputs=[]
        )

        with gr.Tab(label='Video example'):
            gr.Examples(
                examples=["./assets/test.mp4"],
                inputs=[video_input],
            )

    app.launch(debug=True, share=True, max_threads=1)

if __name__ == "__main__":
    initialize_predictor("tiny")
    sam2_in_video()