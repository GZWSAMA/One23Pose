datasets = {}
import numpy as np
from PIL import Image
import dataclasses
import math
import os
import torch
import torch.nn.functional as F

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(config):
    if isinstance(config, str):
        name = config
        config = {}
    else:
        name = config.get('name')

    if name not in datasets:
        raise ValueError(f'Unknown dataset: {name}')

    dataset = datasets[name](config)
    return dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Create output directory
    output_dir = './dataset_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    dataset = NpzDatasetAug(json_files=['./data/objaverse_add/processed/train_3d.jsonl'])
    
    # Create a DataLoader
    batch_size = 4  # You can choose an appropriate batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    for batch_idx, batch_data in enumerate(dataloader):
        target_feats = batch_data['target_feats']
        target_coords = batch_data['target_coords']
        txt = batch_data['txt']
        source = batch_data['source']

        print(f"Batch {batch_idx}:")
        print(f"  Target shape: {target_feats.shape}, {target_coords.shape} (if applicable, depending on the type of target)")
        print(f"  Text (prompt): {txt}")
        print(f"  Text (prompt): {txt}")
        print(f"  Source shape: {source.shape}")

        # You can add more processing or checks here depending on your requirements
        if batch_idx >= 2:  # Just loop through a few batches for testing
            break
            
    # Test single item and save info
    # print(f"Dataset size: {len(dataset)}")
    # with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
    #     f.write(f"Dataset size: {len(dataset)}\n")
        
    #     sample = dataset[0]
    #     f.write("\nSample keys: " + str(sample.keys()) + "\n")
    #     f.write("\nShapes:\n")
    #     for key, value in sample.items():
    #         if isinstance(value, torch.Tensor):
    #             f.write(f"{key}: {value.shape}\n")
        
    
    # # Save sample visualizations
    # for i in range(min(5, len(dataset))):  # Save first 5 samples
    #     sample = dataset[i]
        
    #     plt.figure(figsize=(15, 5))
        
    #     plt.subplot(132)
    #     plt.imshow((sample['source'].numpy() + 1) / 2)
    #     plt.title('Source')
    #     plt.axis('off')
        
    #     plt.savefig(os.path.join(output_dir, f'sample_{i}.png'), 
    #                bbox_inches='tight', 
    #                pad_inches=0.1,
    #                dpi=300)
    #     plt.close()
    
    # print(f"Results saved to {output_dir}")