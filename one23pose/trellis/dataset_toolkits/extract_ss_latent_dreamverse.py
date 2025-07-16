import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import tqdm

import trellis.models as models


torch.set_grad_enabled(False)


def get_voxels(instance):

    position = np.load(os.path.join(opt.output_dir, instance['target_slat']))['coords']
    coords = torch.from_numpy(position[:,1:]).int().contiguous()
    ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--enc_pretrained', type=str, default='weights/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    # parser.add_argument('--dec_pretrained', type=str, default='weights/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
    #                     help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--batch_size', type=str, default=128,
                        help='Batch size to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
        # decoder = models.from_pretrained(opt.dec_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    metadata = []
    with open(os.path.join(opt.output_dir, "train.jsonl"), "r") as f:
        for line in f:
            data_term = json.loads(line)
            metadata += [data_term]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    # metadata = metadata[0:8192]
    records = []
    
    # filter out objects that are already processed
    for data_dict in copy.copy(metadata):
        if os.path.exists(os.path.join(opt.save_dir, os.path.dirname(data_dict['target_slat']), 'ss.npz')):
            records.append({'obj_index': os.path.dirname(data_dict['source_image']), f'ss_latent_{latent_name}': True})
            metadata.remove(data_dict)
    
    def loader(data):
        try:
            ss = get_voxels(data)[None].float()
            load_queue.put((os.path.join(os.path.dirname(instance['target_slat']), 'ss.npz'), ss))
        except Exception as e:
            print(f"Error loading features for {sha256}: {e}")
    
    def saver(save_rel_path, pack):
        save_path = os.path.join(opt.save_dir, save_rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **pack)
        records.append({'obj_index': os.path.dirname(save_rel_path), f'ss_latent_{latent_name}': True})
    
    # encode latents
    num_iterations = len(metadata) // opt.batch_size + 1 if len(metadata) % opt.batch_size != 0 else len(metadata) // opt.batch_size
    for i in tqdm(range(num_iterations)):
        try:
            instance = metadata[i * opt.batch_size: (i + 1) * opt.batch_size]
            ss_batch = []
            for data_dict in instance:
                ss = get_voxels(data_dict)[None].float()
                ss_batch.append(ss)
            ss_batch = torch.cat(ss_batch, dim=0)
            ss_batch = ss_batch.cuda().float()
            latent = encoder(ss_batch, sample_posterior=False)
            # coords = torch.argwhere(decoder(latent)>0)[:, [0, 2, 3, 4]].int()
            assert torch.isfinite(latent).all(), "Non-finite latent"
            for j, data_dict in enumerate(instance):
                pack = {
                    'mean': latent[j].cpu().numpy(),
                }
                saver(os.path.join(os.path.dirname(data_dict['target_slat']).split('/')[-1], 'ss.npz'), pack)        
        except Exception as e:
            continue
            print(f"Error loading features for {i * opt.batch_size + start} to {(i + 1) * opt.batch_size + start}: {e}")
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.save_dir, f'ss_latent_{latent_name}_{opt.rank}.csv'), index=False)
