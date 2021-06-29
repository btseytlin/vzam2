import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from kedro.framework.session import get_current_session
from vzam.indexer import Indexer


def build_index(video_keyframes, parameters):
    session = get_current_session()
    context = session.load_context()
    out_dataset_path = context.catalog.datasets.indexer._filepath
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    indexer = Indexer(**parameters['indexer_params'])

    feature_batches = []
    id_batches = []

    for features_path in tqdm(video_keyframes):
        df = pd.read_csv(features_path)
        df.index = df.time
        df = df.drop(columns=['time'])
        features = df.values.astype(np.float32)

        keyframe_indices = video_keyframes[features_path]['indices']
        keyframe_times = video_keyframes[features_path]['times']
        keyframe_features = features[keyframe_indices]
        keyframe_ids = [dict(time=t, file_path=features_path) for t in keyframe_times]

        feature_batches.append(keyframe_features)
        id_batches.append(keyframe_ids)

    all_features = np.concatenate(feature_batches)
    all_ids = np.concatenate(id_batches).tolist()

    indexer.train(all_features)
    indexer.add(all_features, all_ids)

    return indexer
