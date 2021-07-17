import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from kedro.framework.session import get_current_session
import pickle
from vzam.indexer import Indexer


def train_indexer(indexer, features, labels, clip_times):
    features = np.array(features).astype(np.float32)
    ids = [dict(label=label, clip_time=clip_times) for label, clip_times in zip(labels, clip_times)]
    indexer.train(features)
    indexer.add(features, ids)
    return indexer


def build_index(video_order2_features, parameters):
    indexer = Indexer(**parameters['indexer_params'])

    features = []
    labels = []
    clip_times = []

    for dir_entry in video_order2_features:
        video_clips_dir_path = dir_entry.path
        video_name = dir_entry.name
        for file_entry in os.scandir(video_clips_dir_path):
            path = file_entry.path
            clip_time_min, clip_time_max, feature_vector = pickle.load(open(path, 'rb'))
            features.append(feature_vector)
            labels.append(video_name)
            clip_times.append((clip_time_min, clip_time_max))

    return train_indexer(indexer, features, labels, clip_times)
