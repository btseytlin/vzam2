import os
import logging
import pickle
import numpy as np
from kedro.framework.session import get_current_session
from tqdm.auto import tqdm
from ..get_video_features.nodes import get_frame_preprocessor, extract_video_features
from ...utils import get_feature_extractor


def get_clip_features(video_clips, parameters):
    session = get_current_session()
    context = session.load_context()

    feature_extractor = get_feature_extractor(parameters['feature_extractor'])
    feature_extractor = feature_extractor.to(parameters['device'])

    frame_preprocess = get_frame_preprocessor(parameters)

    out_dataset_path = context.catalog.datasets.clip_features._dir_path
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    for video_dir in tqdm(video_clips):
        for video_file in os.scandir(video_dir.path):
            fpath = video_file.path
            name = video_file.name
            out_fpath = os.path.join(out_dataset_path, name + '.csv')

            if os.path.exists(out_fpath):
                logging.warning('%s features already exist, skipping', out_fpath)
                continue

            times, video_features = extract_video_features(fpath,
                                                           frame_preprocessor=frame_preprocess,
                                                           feature_extractor=feature_extractor,
                                                           fps=parameters['fps'],
                                                           device=parameters['device'])

            with open(out_fpath, 'wb') as f:
                pickle.dump((times, video_features), f)
