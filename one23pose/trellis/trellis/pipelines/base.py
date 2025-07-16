from typing import *
import torch
import torch.nn as nn
from .. import models
import pytorch_lightning as pl
from ..modules import sparse as sp

class Pipeline(pl.LightningModule):
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        super().__init__()  # Initialize the LightningModule
        self.t_scheduler = 'uniform'
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
        if 'slat_flow_model' in self.models:
            self.slat_flow_model = self.models['slat_flow_model']
        if 'sparse_structure_flow_model' in self.models:
            self.sparse_structure_flow_model = self.models['sparse_structure_flow_model']

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {
            k: models.from_pretrained(f"{path}/{v}")
            for k, v in args['models'].items()
        }

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    def load_model(self, name: str, path: str):
        _model = self.models[name]
        states = torch.load(path)
        if 'state_dict' in states:
            states = states['state_dict']
        states = {k.replace(f"{name}.", ""): v for k, v in states.items()}
        _model.load_state_dict(states, False)
        self.models[name] = _model
        
    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        for model in self.models.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def p_losses(self, x_0, cond, t, noise):
        x_t, gt_v = self.slat_sampler._get_model_gt(x_0, t, noise)

        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        pred_v = self(x_t, t, cond)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        loss_simple = self.mse_loss_sparse(pred_v, gt_v)
                
        # Ignore NaN losses
        valid_loss_simple = loss_simple[~torch.isnan(loss_simple)]
        loss_dict.update({f'{prefix}/loss_simple': valid_loss_simple.mean()})
        loss = valid_loss_simple.mean()
        
        return loss, loss_dict
    
    def mse_loss_sparse(self, pred, target):
        diff = pred.feats - target.feats
        return diff ** 2
        
    def training_step(self, batch, batch_idx):
        if self.t_scheduler == 'uniform':
            t = torch.rand(1).item()
        elif type(self.t_scheduler)==float:
            t = self.t_scheduler
        else:
            assert False, 'Not implemented'
            
        targets, cond, noise = self.get_input(batch)
        loss, loss_dict = self.p_losses(targets, cond, t, noise)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        t = 0.9  # Use fixed t=0.5 for validation
        targets, cond, noise = self.get_input(batch)
        loss, loss_dict = self.p_losses(targets, cond, t, noise)
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        raise NotImplementedError