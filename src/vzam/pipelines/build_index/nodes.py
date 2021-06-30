import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from kedro.framework.session import get_current_session
from vzam.indexer import Indexer


def build_index(video_keyframes, video_features, parameters):
    indexer = Indexer(**parameters['indexer_params'])

    feature_batches = []
    id_batches = []

    video_features_name_to_path = {entry.name: entry.path for entry in video_features}

    for name in tqdm(video_keyframes):
        features_path = video_features_name_to_path[name]
        df = pd.read_csv(features_path)
        df.index = df.time
        df = df.drop(columns=['time'])
        features = df.values.astype(np.float32)

        keyframe_indices = video_keyframes[name]['indices']
        keyframe_times = video_keyframes[name]['times']
        keyframe_features = features[keyframe_indices]
        keyframe_ids = [dict(time=t, label=name) for t in keyframe_times]

        feature_batches.append(keyframe_features)
        id_batches.append(keyframe_ids)

    all_features = np.concatenate(feature_batches)
    all_ids = np.concatenate(id_batches).tolist()

    indexer.train(all_features)
    indexer.add(all_features, all_ids)
    assert indexer.index.ntotal == len(all_features) == len(indexer.metadata) == len(all_ids)

    return indexer
