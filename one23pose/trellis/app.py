import gradio as gr
import spaces
import os
os.environ['SPCONV_ALGO'] = 'native'
import numpy as np
import imageio
import uuid
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import trimesh
import torch

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = "./tmp/Trellis-demo"

import os
os.environ["PATH"] += os.path.join(os.getcwd(), "binvox")
os.makedirs(TMP_DIR, exist_ok=True)

def preprocess_mesh(mesh_prompt):
    print("Processing mesh")
    trimesh_mesh = trimesh.load_mesh(mesh_prompt)
    trimesh_mesh.export(mesh_prompt+'.glb')
    return mesh_prompt+'.glb'

def preprocess_image(image):
    image = pipeline.preprocess_image(image, resolution=1024)
    return image

def save_slat(slat, save_path: str):
    """Save SLAT features and coordinates to a npz file."""
    feats_numpy = slat.feats.detach().cpu().numpy()
    coords_numpy = slat.coords.detach().cpu().numpy()

    np.savez(
        save_path,
        feats=feats_numpy,
        coords=coords_numpy
    )
    
@spaces.GPU
def generate_3d(image: Image.Image, neg_image_prompt: Image.Image, mesh_prompt, 
                export_format: str, seed: int = -1,
                ss_guidance_strength: float = 7.5, ss_sampling_steps: int = 12, 
                slat_guidance_strength: float = 15, slat_sampling_steps: int = 25):
    """Generate 3D model and preview video from input image."""
    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)
        
    
    # Handle mesh prompt if provided
    if mesh_prompt is not None:
        trimesh_mesh = trimesh.load_mesh(mesh_prompt)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix

        # Convert scene to mesh if necessary
        if hasattr(trimesh_mesh, 'geometry'):
            # If it's a scene, get the first mesh
            # Assuming the scene has at least one mesh
            mesh_name = list(trimesh_mesh.geometry.keys())[0]
            trimesh_mesh = trimesh_mesh.geometry[mesh_name]
        trimesh_mesh = postprocessing_utils.normalize_trimesh(trimesh_mesh)
        trimesh_mesh.apply_transform(transformation_matrix)

        outputs = pipeline.run(
            image,
            ref_image=neg_image_prompt,
            init_mesh=trimesh_mesh,
            seed=seed,
            formats=["mesh", "gaussian"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        generated_mesh = outputs['mesh'][0]
        generated_gs = outputs['gaussian'][0]
        
    else:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["mesh", "gaussian"],
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        generated_mesh = outputs['mesh'][0]
        generated_gs = outputs['gaussian'][0]
        
    # Save video and mesh
    output_id = str(uuid.uuid4())
    video_path = f"{TMP_DIR}/{output_id}_preview.mp4"
    mesh_path = f"{TMP_DIR}/{output_id}.{export_format}"
    gs_path = f"{TMP_DIR}/{output_id}.ply"
    slat_path = f"{TMP_DIR}/{output_id}.npz"
    generated_slat = outputs['slat'][0]
    
    save_slat(generated_slat, slat_path)
    # Save video
    video_geo = render_utils.render_video(generated_mesh, resolution=1024, num_frames=120)['color']
    imageio.mimsave(video_path, video_geo, fps=15)

    generated_gs = generated_gs.save_ply(gs_path)
    
    # Export mesh in selected format
    trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
    trimesh_mesh.export(mesh_path)

    return video_path, gs_path, slat_path

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # Input controls
            image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil")
            neg_image_prompt = gr.Image(label="Negative Image Prompt", image_mode="RGBA", type="pil")
            mesh_prompt = gr.Model3D(label="Model Prompt (Optional)")

            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(-1, MAX_SEED, label="Seed", value=0, step=1)
                export_format = gr.Dropdown(
                    choices=["obj", "glb", "ply"],
                    value="glb",
                    label="Export Format",
                    info="Choose the format for the exported 3D model"
                )
                
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=50, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

            generate_btn = gr.Button("Generate 3D Model")

        with gr.Column():
            # Output displays
            video_output = gr.Video(label="Preview", autoplay=True, loop=True)
            model_output = gr.Model3D(label="3D Model (Mesh)")
            slat_output = gr.File(label="SLAT Features", file_count="single")
            
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt]
    )
    neg_image_prompt.upload(
        preprocess_image,
        inputs=[neg_image_prompt],
        outputs=[neg_image_prompt]
    )
    mesh_prompt.upload(
        preprocess_mesh,
        inputs=[mesh_prompt],
        outputs=[mesh_prompt],
    )
    generate_btn.click(
        generate_3d,
        inputs=[
            image_prompt, neg_image_prompt, mesh_prompt, export_format, seed,
            ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps
        ],
        outputs=[video_output, model_output, slat_output]
    )

if __name__ == "__main__":
    from peft import LoraConfig, get_peft_model 
    # peft_config = LoraConfig(
    #     r=512,
    #     lora_alpha=512,
    #     lora_dropout=0.0,
    #     target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
    # )
    # from peft import LoraConfig, get_peft_model
    # pipeline = TrellisImageTo3DPipeline.from_pretrained("jetx/TRELLIS-image-large")
    # pipeline.slat_flow_model = get_peft_model(pipeline.models['slat_flow_model'], peft_config)
    # pipeline.slat_flow_model.print_trainable_parameters()
    # pipeline.models['slat_flow_model'] = pipeline.slat_flow_model
    # pipeline.load_model("slat_flow_model", "/baai-cwm-vepfs/cwm/zheng.geng/code/pose/One23Pose/checkpoints/Stable3DGen/epoch=49-step=123100.ckpt")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("/baai-cwm-vepfs/cwm/zheng.geng/code/pose/One23Pose/checkpoints/Trellis")
    pipeline.cuda()
    demo.launch(share=False, server_name="0.0.0.0", server_port=9944, auth=("bbcc", "bbcc"))