import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

from .base import BaseDataset

class DreamVerseDataset(BaseDataset):
    def collate_fn(self, batch):
        batched_condition_feats = []
        batched_condition_coords = []
        batched_target_feats = []
        batched_target_coords = []
        other_keys = {}

        for idx, sample in enumerate(batch):
            condition_feats = sample['conditional_slat']['feats']
            condition_coords = sample['conditional_slat']['coords'][..., 1:]
            target_feats = sample['target_slat']['feats']
            target_coords = sample['target_slat']['coords'][..., 1:]

            batch_coords = torch.full((condition_coords.shape[0], 1), idx, dtype=condition_coords.dtype)

            batched_condition_feats.append(condition_feats)
            batched_condition_coords.append(torch.cat([batch_coords, condition_coords], dim=1))
            batched_target_feats.append(target_feats)
            batched_target_coords.append(torch.cat([batch_coords, target_coords], dim=1))

            for key, value in sample.items():
                if not key.endswith('slat'):
                    if key not in other_keys:
                        other_keys[key] = []
                    other_keys[key].append(value)

        batched_condition_feats = torch.cat(batched_condition_feats, dim=0)
        batched_condition_coords = torch.cat(batched_condition_coords, dim=0)
        batched_target_feats = torch.cat(batched_target_feats, dim=0)
        batched_target_coords = torch.cat(batched_target_coords, dim=0)

        batched_data = {
            'conditional_slat': {
                'feats': batched_condition_feats,
                'coords': batched_condition_coords
            },
            'target_slat': {
                'feats': batched_target_feats,
                'coords': batched_target_coords
            }
        }

        for key, value in other_keys.items():
            batched_data[key] = default_collate(value)

        return batched_data

@register("DreamVerse-datamodule")
class DreamVerseDataModule(pl.LightningDataModule):
    def __init__(self, cfg = None) -> None:
        super().__init__()

    def setup(self, data_dirs, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = DreamVerseDataset(data_dirs, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DreamVerseDataset(data_dirs, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = DreamVerseDataset(data_dirs, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

