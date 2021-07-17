from argparse import Namespace

import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
import timm
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_metric_learning import losses
from vzam.utils import l_normalize


def dfs_freeze(model, unfreeze=False):
    for param in model.parameters():
        param.requires_grad = unfreeze

    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = unfreeze
        dfs_freeze(child, unfreeze=unfreeze)


class Order2Extractor(pl.LightningModule):
    def __init__(self, hparams = None, only_train_layers=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.only_train_layers = only_train_layers

        self.loss = losses.ContrastiveLoss(pos_margin=hparams.pos_margin, neg_margin=hparams.neg_margin)
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(3, 6, kernel_size=3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d(300),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d(50),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(20),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
        )

        # Freeze layers that dont require grad
        if only_train_layers:
            dfs_freeze(self.trunk)

            for layer_name_or_getter in only_train_layers:
                if isinstance(layer_name_or_getter, str):
                    layer = getattr(self.trunk, layer_name_or_getter)

                else:
                    layer = layer_name_or_getter(self.trunk)
                dfs_freeze(layer, unfreeze=True)

    def forward(self, x):
        return l_normalize(self.trunk(x))

    def predict_proba(self, x):
        probabilities = nn.functional.softmax(self.forward(x), dim=1)
        return probabilities

    def predict(self, x):
        return torch.max(self.forward(x), 1)[1]

    def configure_optimizers(self):
        trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=self.hparams.lr or self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        return (
            [optimizer],
            []
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, 1)
        embeddings = self(x)
        loss = self.loss(embeddings, y)
        return loss
