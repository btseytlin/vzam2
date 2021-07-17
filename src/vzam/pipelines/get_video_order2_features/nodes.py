import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from kedro.framework.session import get_current_session
from tqdm.auto import tqdm
from ..train_order2_extractor.nodes import get_dataset


def extract_order2_features(video_features_dir_entries, order2_extractor, parameters):
    video_clips_dataset = get_dataset(video_features_dir_entries,
                                      video_clip_size=parameters['video_clip_size'],
                                      minimal_clip_size=parameters['minimal_clip_size'])

    all_clip_source_names, all_clip_times, all_clip_order2_features = [], [], []
    order2_extractor.to(parameters['device'])
    with torch.no_grad():
        for i, (x, _) in tqdm(enumerate(video_clips_dataset)):
            if len(x) < parameters['minimal_clip_size']:
                continue
            x = x.unsqueeze(0).unsqueeze(1).to(parameters['device'])
            clip_order2_features = order2_extractor(x).cpu().numpy()
            source_name = video_clips_dataset.source_video_names[i]
            clip_times = video_clips_dataset.clip_times[i]

            all_clip_source_names.append(source_name)
            all_clip_times.append(clip_times)
            all_clip_order2_features.append(clip_order2_features)

    all_clip_order2_features = np.concatenate(all_clip_order2_features)

    return all_clip_source_names, all_clip_times, all_clip_order2_features


def get_video_order2_features(video_features, order2_extractor, parameters):
    session = get_current_session()
    context = session.load_context()

    out_dataset_path = context.catalog.datasets.video_order2_features._dir_path
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    all_clip_source_names, all_clip_times, all_clip_order2_features = extract_order2_features(video_features, order2_extractor, parameters)

    for basename, clip_times, clip_order2_features in zip(all_clip_source_names, all_clip_times, all_clip_order2_features):
        clip_fname = f'{basename}_{int(clip_times[0])}_{int(clip_times[-1])}.pkl'
        clip_dir = os.path.join(out_dataset_path, basename)
        if not os.path.exists(clip_dir):
            os.makedirs(clip_dir)
        clip_fpath = os.path.join(clip_dir, clip_fname)

        with open(clip_fpath, 'wb') as f:
            pickle.dump((clip_times[0], clip_times[-1], clip_order2_features), f)
