import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from vzam.utils import get_feature_extractor


def get_video_order2_features(video_features, order2_extractor, parameters):
    order2_extractor = order2_extractor.to(parameters['device'])

    preprocessor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    order2_features = {}
    for dir_entry in video_features:
        path = dir_entry.path
        name = dir_entry.name

        df = pd.read_csv(path)
        df.index = df.time
        df = df.drop(columns=['time'])
        features = df.values

        features_tensor = preprocessor(features).float().unsqueeze(dim=0).to(parameters['device'])

        order2_features[name] = order2_extractor(features_tensor)[0].tolist()

    return order2_features
