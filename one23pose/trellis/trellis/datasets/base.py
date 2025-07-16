import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import random
import pdb
import PIL.Image as Image
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
from trellis.modules import sparse as sp
from torch.utils.data import DataLoader

class BaseDataset(Dataset):
    VALID_ASSETS = [
        'normal', 
        'depth', 
        'albedo', 
        'lighting', 
        'mask']
    
    def __init__(self, 
                 data_dirs=None,
                 split='train',
                 source_aug=None,
                 resolution=518, 
                 task='normal'):
        self.data = []
        for data_dir in data_dirs:
            json_file = os.path.join(data_dir, f'{split}.jsonl')
            with open(json_file, "r") as f:
                for line in f:
                    data_term = json.loads(line)
                    data_term['data_dir'] = data_dir
                    self.data += [data_term]
        self.to_Tensor = T.ToTensor()
        self.source_aug = source_aug
        self.task = task
        self.resolution = resolution
        self.resize_transform = T.Resize(self.resolution)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception as e:
            random_idx = random.randint(0, len(self.data) - 1)
            return self._load_item(random_idx)

    def _load_item(self, idx):
        item = self.data[idx]
        data_dir = item['data_dir']
        data_dict = {}
        
        if 'source_slat' in item:
            source_slat = np.load(os.path.join(data_dir, item['source_slat']))
            data_dict['source_slat'] = {key: torch.from_numpy(source_slat[key]) for key in source_slat.keys()}
        
        if 'target_slat' in item:
            target_slat = np.load(os.path.join(data_dir, item['target_slat']))
            data_dict['target_slat'] = {key: torch.from_numpy(target_slat[key]) for key in target_slat.keys()}
        
        if 'source_image' in item:
            if item['source_image'].endswith('.npz'):
                source_image = np.load(os.path.join(data_dir, item['source_image']))[self.task]
                if source_image.dtype == np.uint8:
                    source_image = source_image.astype(np.float32) / 255.0
                source_image = torch.tensor(source_image).permute(0, 3, 1, 2)
                source_image_path = item['source_image']
            elif item['source_image'].endswith('.png') or item['source_image'].endswith('.jpg'):
                source_image = Image.open(os.path.join(data_dir, item['source_image']))
                source_image /= 255.0
                source_image = self.to_Tensor(source_image).unsqueeze(0).float()
            
            if self.source_aug is not None:
                source_image = self.source_aug(source_image)
            data_dict['source_image'] = self.resize_transform(source_image)
            
        if 'reference_image' in item:
            if item['reference_image'].endswith('.npz'):
                reference_image = np.load(os.path.join(data_dir, item['reference_image']))[self.task]
                reference_image = torch.tensor(reference_image).float().permute(0, 3, 1, 2)
            elif item['reference_image'].endswith('.png') or item['reference_image'].endswith('.jpg'):
                reference_image = Image.open(os.path.join(data_dir, item['reference_image']))
                reference_image = self.to_Tensor(reference_image).unsqueeze(0).float()
            if reference_image.max() > 1:
                reference_image /= 255.0
            data_dict['reference_image'] = self.resize_transform(reference_image)
        
        return data_dict

def custom_collate(batch):
    """
    Custom collate function that handles sparse tensors along with other batch elements.
    
    Args:
        batch: List of dictionaries containing 'target_feats', 'target_coords', 'condition_feats', 'condition_coords', 'source' and 'ref'
        
    Returns:
        Dictionary with batched data
    """
    # Initialize lists to hold the batched data
    batched_feats = []
    batched_coords = []
    batched_condition_feats = []
    batched_condition_coords = []
    batched_source = []
    batched_ref = []
    
    batch_size = len(batch)
    for idx, sample in enumerate(batch):
        # Handle target sparse tensor components
        feats = sample['target_feats']
        coords = sample['target_coords'][..., 1:]
        
        # Add batch dimension to coordinates
        batch_coords = torch.full((coords.shape[0], 1), idx, dtype=coords.dtype)
        
        batched_feats.append(feats)
        batched_coords.append(torch.cat([batch_coords, coords], dim=1))
        
        # Handle condition sparse tensor components
        condition_feats = sample['condition_feats']
        condition_coords = sample['condition_coords'][..., 1:]

        batched_condition_feats.append(condition_feats)
        batched_condition_coords.append(torch.cat([batch_coords, condition_coords], dim=1))
        
        # Handle other batch elements
        batched_source.append(sample['source'])
        batched_ref.append(sample['ref'])
    
    # Concatenate target features and coordinates
    batched_feats = torch.cat(batched_feats, dim=0)
    batched_coords = torch.cat(batched_coords, dim=0)
    
    # Concatenate condition features and coordinates
    batched_condition_feats = torch.cat(batched_condition_feats, dim=0)
    batched_condition_coords = torch.cat(batched_condition_coords, dim=0)
    
    # Stack source tensors
    batched_source = torch.stack(batched_source, dim=0)
    batched_ref = torch.stack(batched_ref, dim=0)
    
    return {
        'target_feats': batched_feats,
        'target_coords': batched_coords,
        'condition_feats': batched_condition_feats,
        'condition_coords': batched_condition_coords,
        'source': batched_source,
        'ref': batched_ref
    }
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Create output directory
    output_dir = './dataset_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = BaseDataset(json_files=['./data/train.jsonl'])
    
    # Create a DataLoader
    # batch_size = 4  # You can choose an appropriate batch size
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
            
    # Test single item and save info
    print(f"Dataset size: {len(dataset)}")
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
        f.write(f"Dataset size: {len(dataset)}\n")
        
        sample = dataset[0]
        f.write("\nSample keys: " + str(sample.keys()) + "\n")
        f.write("\nShapes:\n")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                f.write(f"{key}: {value.shape}\n")
        
    
    # Save sample visualizations
    for i in range(min(5, len(dataset))):  # Save first 5 samples
        fig = plt.figure(figsize=(15, 5))
        sample = dataset[i]
        
        plt.subplot(132)
        plt.imshow(sample['source_image'][0].permute(1, 2, 0).numpy())
        plt.title('Source')
        plt.axis('off')
        
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(133, projection='3d')
        coords = sample['target_slat']['coords'][..., 1:].numpy() # [N, 3]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)
        ax.set_xlim([0, 63])
        ax.set_ylim([0, 63])
        ax.set_zlim([0, 63])
        
        plt.savefig(os.path.join(output_dir, f'sample_{i}.png'), 
                   bbox_inches='tight', 
                   pad_inches=0.1,
                   dpi=300)
        plt.close(fig)
    
    print(f"Results saved to {output_dir}")