import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from kedro.framework.session import get_current_session
import pickle
from vzam.indexer import Indexer


def build_index(video_order2_features, parameters):
    indexer = Indexer(**parameters['indexer_params'])

    features = []
    ids = []

    for dir_entry in video_order2_features:
        video_clips_dir_path = dir_entry.path
        video_name = dir_entry.name
        for file_entry in os.scandir(video_clips_dir_path):
            path = file_entry.path
            basename = video_name.split(".")[0]
            clip_time_min, clip_time_max, feature_vector = pickle.load(open(path, 'rb'))
            features.append(feature_vector)
            ids.append({'label': basename, 'clip_time': [clip_time_min, clip_time_max]})


    features = np.array(features).astype(np.float32)

    indexer.train(features)
    indexer.add(features, ids)
    assert indexer.index.ntotal == len(features) == len(indexer.metadata) == len(ids)

    return indexer
