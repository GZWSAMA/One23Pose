import gc

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

# Try to import HF SAM support
try:
    from app_3rd.sam_utils.hf_sam_predictor import get_hf_sam_predictor, HFSamPredictor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

models = {
  'vit_b': 'app_3rd/sam_utils/checkpoints/sam_vit_b_01ec64.pth',
  'vit_l': 'app_3rd/sam_utils/checkpoints/sam_vit_l_0b3195.pth',
  'vit_h': 'app_3rd/sam_utils/checkpoints/sam_vit_h_4b8939.pth'
}


def get_sam_predictor(model_type='vit_b', device=None, image=None, use_hf=True, predictor=None):
  """
  Get SAM predictor with option to use HuggingFace version
  
  Args:
      model_type: Model type ('vit_b', 'vit_l', 'vit_h')
      device: Device to run on
      image: Optional image to set immediately
      use_hf: Whether to use HuggingFace SAM instead of original SAM
  """
  if predictor is not None:
    return predictor
  if use_hf:
    if not HF_AVAILABLE:
      raise ImportError("HuggingFace SAM not available. Install transformers and huggingface_hub.")
    return get_hf_sam_predictor(model_type, device, image)
  
  # Original SAM logic
  if device is None and torch.cuda.is_available():
    device = 'cuda'
  elif device is None:
    device = 'cpu'
  # sam model
  sam = sam_model_registry[model_type](checkpoint=models[model_type])
  sam = sam.to(device)

  predictor = SamPredictor(sam)
  if image is not None:
    predictor.set_image(image)
  return predictor


def run_inference(predictor, input_x, selected_points, multi_object: bool = False):
  """
  Run inference with either original SAM or HF SAM predictor
  
  Args:
      predictor: SamPredictor or HFSamPredictor instance
      input_x: Input image
      selected_points: List of (point, label) tuples
      multi_object: Whether to handle multiple objects
  """
  if len(selected_points) == 0:
    return []
  
  # Check if using HF SAM
  if isinstance(predictor, HFSamPredictor):
    return _run_hf_inference(predictor, input_x, selected_points, multi_object)
  else:
    return _run_original_inference(predictor, input_x, selected_points, multi_object)


def _run_original_inference(predictor: SamPredictor, input_x, selected_points, multi_object: bool = False):
  """Run inference with original SAM"""
  points = torch.Tensor(
      [p for p, _ in selected_points]
  ).to(predictor.device).unsqueeze(1)

  labels = torch.Tensor(
      [int(l) for _, l in selected_points]
  ).to(predictor.device).unsqueeze(1)

  transformed_points = predictor.transform.apply_coords_torch(
      points, input_x.shape[:2])

  masks, scores, logits = predictor.predict_torch(
    point_coords=transformed_points[:,0][None],
    point_labels=labels[:,0][None],
    multimask_output=False,
  )
  masks = masks[0].cpu().numpy()  # N 1 H W   N is the number of points

  gc.collect()
  torch.cuda.empty_cache()

  return [(masks, 'final_mask')]


def _run_hf_inference(predictor: HFSamPredictor, input_x, selected_points, multi_object: bool = False):
  """Run inference with HF SAM"""
  # Prepare points and labels for HF SAM
  select_pts = [[list(p) for p, _ in selected_points]]
  select_lbls = [[int(l) for _, l in selected_points]]
  
  # Preprocess inputs
  inputs = predictor.preprocess(input_x, select_pts, select_lbls)

  # Run inference
  with torch.no_grad():
    outputs = predictor.model(**inputs)
  
  # Post-process masks
  masks = predictor.processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu(),
  )
  masks = masks[0][:,:1,...].cpu().numpy()

  gc.collect()
  torch.cuda.empty_cache()

  return [(masks, 'final_mask')]