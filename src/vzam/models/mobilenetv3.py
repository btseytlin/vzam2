import pytorch_lightning as pl
import torch
from torch import nn
import timm


class MobileNetv3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('mobilenetv3_rw', pretrained=True, num_classes=0)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
