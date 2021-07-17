import os
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from kedro.framework.session import get_current_session
from tqdm.auto import tqdm
from ..train_order2_extractor.nodes import get_dataset


def get_video_order2_features(video_features, order2_extractor, parameters):
    session = get_current_session()
    context = session.load_context()

    out_dataset_path = context.catalog.datasets.video_order2_features._dir_path
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    order2_extractor = order2_extractor.to(parameters['device'])

    video_clips_dataset = get_dataset(video_features, video_clip_size=parameters['video_clip_size'])
    batch_size = parameters['batch_size']
    video_clips_loader = DataLoader(video_clips_dataset,
                                    batch_size=batch_size,
                                    num_workers=parameters['data_loader_workers'],
                                    shuffle=False)

    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(video_clips_loader)):
            indices = range(i*batch_size, i*batch_size+batch_size)
            x = torch.unsqueeze(x, 1).to(parameters['device'])
            batch_clip_order2_features = order2_extractor(x).cpu().numpy()
            batch_source_names = video_clips_dataset.source_video_names[indices]
            batch_clip_times = video_clips_dataset.clip_times[indices]

            for j in range(len(batch_source_names)):
                basename = batch_source_names[j]
                clip_times = batch_clip_times[j]
                clip_order2_features = batch_clip_order2_features[j]

                clip_fname = f'{basename}_{int(clip_times[0])}_{int(clip_times[-1])}.pkl'
                clip_dir = os.path.join(out_dataset_path, basename)
                if not os.path.exists(clip_dir):
                    os.makedirs(clip_dir)
                clip_fpath = os.path.join(clip_dir, clip_fname)

                with open(clip_fpath, 'wb') as f:
                    pickle.dump((clip_times[0], clip_times[-1], clip_order2_features), f)

