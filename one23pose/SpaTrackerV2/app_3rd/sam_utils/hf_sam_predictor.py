import gc
import numpy as np
import torch
from typing import Optional, Tuple, List, Union
import warnings
import cv2
try:
    from transformers import SamModel, SamProcessor
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("transformers or huggingface_hub not available. HF SAM models will not work.")

# Hugging Face model mapping
HF_MODELS = {
    'vit_b': 'facebook/sam-vit-base',
    'vit_l': 'facebook/sam-vit-large', 
    'vit_h': 'facebook/sam-vit-huge'
}

class HFSamPredictor:
    """
    Hugging Face version of SamPredictor that wraps the transformers SAM models.
    This class provides the same interface as the original SamPredictor for seamless integration.
    """
    
    def __init__(self, model: SamModel, processor: SamProcessor, device: Optional[str] = None):
        """
        Initialize the HF SAM predictor.
        
        Args:
            model: The SAM model from transformers
            processor: The SAM processor from transformers
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        self.model = model
        self.processor = processor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Store the current image and its features
        self.original_size = None
        self.input_size = None
        self.features = None
        self.image = None

    @classmethod
    def from_pretrained(cls, model_name: str, device: Optional[str] = None) -> 'HFSamPredictor':
        """
        Load a SAM model from Hugging Face Hub.
        
        Args:
            model_name: Model name from HF_MODELS or direct HF model path
            device: Device to load the model on
        
        Returns:
            HFSamPredictor instance
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers and huggingface_hub are required for HF SAM models")
        
        # Map model type to HF model name if needed
        if model_name in HF_MODELS:
            model_name = HF_MODELS[model_name]
        
        print(f"Loading SAM model from Hugging Face: {model_name}")
        
        # Load model and processor
        model = SamModel.from_pretrained(model_name)
        processor = SamProcessor.from_pretrained(model_name)
        return cls(model, processor, device)
    
    def preprocess(self, image: np.ndarray,
                         input_points: List[List[float]], input_labels: List[int]) -> None:
        """
        Set the image for prediction. This preprocesses the image and extracts features.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        self.image = image
        self.original_size = image.shape[:2]

        # Use dummy point to ensure processor returns original_sizes & reshaped_input_sizes
        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.input_size = inputs['pixel_values'].shape[-2:]
        self.features = inputs
        return inputs
    

def get_hf_sam_predictor(model_type: str = 'vit_h', device: Optional[str] = None, 
                        image: Optional[np.ndarray] = None) -> HFSamPredictor:
    """
    Get a Hugging Face SAM predictor with the same interface as the original get_sam_predictor.
    
    Args:
        model_type: Model type ('vit_b', 'vit_l', 'vit_h')
        device: Device to run the model on
        image: Optional image to set immediately
    
    Returns:
        HFSamPredictor instance
    """
    if not HF_AVAILABLE:
        raise ImportError("transformers and huggingface_hub are required for HF SAM models")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the predictor
    predictor = HFSamPredictor.from_pretrained(model_type, device)
    
    # Set image if provided
    if image is not None:
        predictor.set_image(image)
    
    return predictor 