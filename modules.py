import torch.nn.functional as F
from utils import compute_eer
import lightning as l
import torch.nn as nn
import numpy as np
import torch


class SpoofModule(l.LightningModule):
    
    def __init__(self, model: nn.Module, scale: float, margin: float) -> None:
        super(SpoofModule, self).__init__()
        
        self.model = model
        self.s = scale
        self.m = margin
        
        self.train_outputs = []
        self.val_outputs = {
            "target": [],
            "prob": []
        }
        
    def set_training_settings(self, optimizer_cfg, scheduler_cfg):
        
        try:
            optimizer = getattr(torch.optim, optimizer_cfg["type"])
        except:
            optimizer = eval(optimizer_cfg["type"])
        self._optimizer = lambda x: optimizer(x, **optimizer_cfg["params"])
        self._scheduler = lambda x: getattr(torch.optim.lr_scheduler, scheduler_cfg["type"])(x, **scheduler_cfg["params"])
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        logits = self.model.forward(x)
        if self.training:
            return logits
        return F.softmax(logits * self.s, dim=-1)[:, 1]
    
    def training_step(self, batch):

        x, y = batch
        y = y.long()
        logits = self.forward(x)
        margin = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), self.m)
        logits = self.s * (logits - margin)
        loss = F.cross_entropy(logits, y)
        
        self.log("LogLoss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):

        x, y = batch
        probs = self.forward(x)
        probs = probs.cpu().numpy()
        target = y.cpu().numpy()
        self.val_outputs["target"].append(target)
        self.val_outputs["prob"].append(probs)
        
    def on_validation_epoch_end(self):

        probs = np.concatenate(self.val_outputs["prob"])
        targets = np.concatenate(self.val_outputs["target"])

        bonafide_idxs = np.where(targets > 0.5)
        spoof_idxs = np.where(targets < 0.5)
        bonafide_probs = probs[bonafide_idxs]
        spoof_probs = probs[spoof_idxs]
        
        self.log("EER", compute_eer(bonafide_probs, spoof_probs)[0] * 100)
        self.val_outputs["target"].clear()
        self.val_outputs["prob"].clear()
        
    def configure_optimizers(self):
        
        optimizer = self._optimizer(self.parameters())
        scheduler = self._scheduler(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }