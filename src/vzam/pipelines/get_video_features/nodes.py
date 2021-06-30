import os
import logging
import torch
import numpy as np
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from torchvision import transforms
from vzam.utils import get_feature_extractor
from PIL import Image
from tqdm.auto import tqdm
from kedro.framework.session import get_current_session


def l_normalize(x, p=2):
    norm = x.norm(p=p, dim=1, keepdim=True)
    x_normalized = x.div(norm.expand_as(x))
    return x_normalized


def get_frame_preprocessor(parameters):
    return transforms.Compose(
        [
            transforms.Resize(parameters['resize_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

def extract_video_features(fpath, frame_preprocessor, feature_extractor, device, fps=5, batch_size=32):
    def process_batch(tensor_list):
        batch = torch.cat(tensor_list)
        features = l_normalize(feature_extractor(batch.to(device))).cpu()
        return features

    try:
        clip = VideoFileClip(fpath)
        nframes = int(fps * clip.duration)
        frames = clip.iter_frames(with_times=True, dtype='uint8', fps=fps)

        vector_batches = []
        batch_tensors = []
        times = []

        for i, (ts, frame_img) in tqdm(enumerate(frames), total=nframes):
            times.append(ts)
            batch_tensors.append(frame_preprocessor(Image.fromarray(frame_img)).unsqueeze(dim=0))

            if len(batch_tensors) > batch_size:
                vector_batches.append(process_batch(batch_tensors))
                batch_tensors = []

        if batch_tensors:
            vector_batches.append(process_batch(batch_tensors))

        vectors = torch.cat(vector_batches)
    finally:
        clip.close()

    return np.array(times), vectors


def get_video_features(train_videos, parameters):
    session = get_current_session()
    context = session.load_context()

    feature_extractor = get_feature_extractor(parameters['feature_extractor'])
    feature_extractor = feature_extractor.to(parameters['device'])

    frame_preprocess = get_frame_preprocessor(parameters)

    out_dataset_path = context.catalog.datasets.video_features._dir_path
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    for video_file in tqdm(train_videos):
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

        features_df = pd.DataFrame(np.array(video_features), index=times)
        features_df.index.name = 'time'

        features_df.to_csv(out_fpath)



