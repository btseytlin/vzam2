
from argparse import Namespace

import torch
import pickle
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from sklearn.preprocessing import LabelEncoder
from vzam.models.order2_extractor import Order2Extractor

from vzam.utils import fname


class VideoClipsDataset(Dataset):
    def __init__(self, clip_features, source_video_names, clip_times, transform=None, *args, **kwargs):
        super().__init__()
        self.clip_features = clip_features
        self.source_video_names = np.array(source_video_names)
        self.clip_times = np.array(clip_times)
        self.transform = transform

        self.y_label_encoder = LabelEncoder()
        self.y = self.y_label_encoder.fit_transform(source_video_names)

    def __getitem__(self, index):
        x = self.transform(self.clip_features[index])
        y = self.y[index]

        return x, y

    def __len__(self):
        return len(self.clip_features)


def extract_video_clip_features(video_features_path, video_clip_size, minimal_clip_size):
    video_clip_features, video_clip_times = [], []

    times, features = pickle.load(open(video_features_path, 'rb'))
    clip_size = video_clip_size
    iterator = range(0, len(features), clip_size) if len(features) > clip_size else [0]
    for i in iterator:
        clip_times = times[i:i + clip_size]
        clip_times = (clip_times.min(), clip_times.max())
        clip_features = features[i:i + clip_size]

        if len(clip_features) < minimal_clip_size:
            continue

        video_clip_features.append(clip_features)
        video_clip_times.append(clip_times)
    return video_clip_features, video_clip_times


def get_dataset(features_entries, video_clip_size, minimal_clip_size):
    preprocessor = transforms.Compose([
    ])

    all_clip_features, all_source_video_names, all_clip_times = [], [], []
    for dir_entry in features_entries:
        path = dir_entry.path
        name = dir_entry.name
        basename = fname(dir_entry.name)

        video_clip_features, video_clip_times = extract_video_clip_features(path, video_clip_size, minimal_clip_size)
        all_clip_features += video_clip_features
        all_clip_times += video_clip_times
        all_source_video_names += [basename] * len(video_clip_features)

    return VideoClipsDataset(all_clip_features, all_source_video_names, all_clip_times, transform=preprocessor)


def train_order2_extractor(video_features, parameters):

    def collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        return [data, target]


    video_clips_dataset = get_dataset(video_features,
                                      video_clip_size=parameters['video_clip_size'],
                                      minimal_clip_size=parameters['minimal_clip_size'])

    video_clips_loader = DataLoader(video_clips_dataset,
                              batch_size=parameters['batch_size'],
                              num_workers=parameters['data_loader_workers'],
                              shuffle=True,
                              collate_fn=collate,
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
    trainer.fit(model, video_clips_loader)
    return model
