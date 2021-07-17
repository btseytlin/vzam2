import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from kedro.framework.session import get_current_session
from tqdm.auto import tqdm
from ..get_video_order2_features.nodes import extract_order2_features


def get_clip_order2_features(clip_features, order2_extractor, parameters):
    session = get_current_session()
    context = session.load_context()

    out_dataset_path = context.catalog.datasets.clip_order2_features._dir_path
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    all_clip_source_names, all_clip_times, all_clip_order2_features = extract_order2_features(clip_features, order2_extractor, parameters)

    for basename, clip_times, clip_order2_features in zip(all_clip_source_names, all_clip_times, all_clip_order2_features):
        clip_fname = f'{basename}_{int(clip_times[0])}_{int(clip_times[-1])}.pkl'
        clip_dir = os.path.join(out_dataset_path, basename)
        if not os.path.exists(clip_dir):
            os.makedirs(clip_dir)
        clip_fpath = os.path.join(clip_dir, clip_fname)

        with open(clip_fpath, 'wb') as f:
            pickle.dump((clip_times[0], clip_times[-1], clip_order2_features), f)
