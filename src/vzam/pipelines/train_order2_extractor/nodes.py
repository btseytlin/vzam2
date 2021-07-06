
from argparse import Namespace

import torch
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from vzam.models.order2_extractor import Order2Extractor


class CustomDataset(Dataset):
    def __init__(self, dict, transform=None, *args, **kwargs):
        super().__init__()
        self.data = np.array(list(dict.values()))
        self.target = np.array(list(dict.keys()))
        self.transform = transform

    def __getitem__(self, index):
        x = self.transform(self.data[index])
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)


def get_dataset(features_entries, is_clips=False):
    preprocessor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    out_dict = {}
    for dir_entry in features_entries:
        path = dir_entry.path
        name = dir_entry.name

        df = pd.read_csv(path)
        df.index = df.time
        df = df.drop(columns=['time'])
        features = df.values

        if is_clips:
            name = name.split('__')[1]
        out_dict[name] = features

    return CustomDataset(out_dict, transform=preprocessor)


def train_order2_extractor(video_features, clip_features, parameters):
    video_dataset = get_dataset(video_features)
    clip_dataset = get_dataset(clip_features, is_clips=True)

    video_loader = DataLoader(video_dataset,
                              batch_size=parameters['batch_size'],
                              num_workers=parameters['data_loader_workers'],
                              shuffle=True,
                              pin_memory=True)

    clip_loader = DataLoader(clip_dataset,
                              batch_size=parameters['batch_size'],
                              num_workers=parameters['data_loader_workers'],
                              shuffle=True,
                              pin_memory=True)

    hparams = Namespace(**parameters['order2_extractor'])

    trainer = Trainer.from_argparse_args(
        hparams,
        reload_dataloaders_every_epoch=True,
        terminate_on_nan=True,
        logger=None,
    )

    # Model
    model = Order2Extractor(hparams=hparams)
    # Training
    trainer.fit(model, dict(video=video_loader, clip=clip_loader))
    return model
