from argparse import Namespace

import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
import timm
import pytorch_lightning as pl
import torch.nn.functional as F


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
        return self.trunk(x)

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
        running_loss = 0
        for video_x, video_y in zip(batch['video'][0], batch['video'][1]):
            for clip_x, clip_y in zip(batch['clip'][0], batch['clip'][1]):

                video_features = self(video_x.unsqueeze(0).float())
                clip_features = self(clip_x.unsqueeze(0).float())

                same_label = video_x == clip_y

                dist = torch.cdist(video_features, clip_features, p=2.0)

                if same_label:
                    running_loss += torch.clamp(dist - self.hparams.pos_margin, min=0)
                else:
                    running_loss += torch.clamp(self.hparams.neg_margin - dist, min=0)
        return running_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)