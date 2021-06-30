import os
import random

from kedro.framework.session import get_current_session
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.resize import resize


def cut_videos_into_clips(train_videos, parameters):
    session = get_current_session()
    context = session.load_context()
    out_dataset_path = context.catalog.datasets.video_clips._dir_path
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    for video_dir_enty in train_videos:
        fpath = video_dir_enty.path
        name = video_dir_enty.name

        video = VideoFileClip(fpath)
        try:
            clip_dir_path = os.path.join(out_dataset_path, name)
            if not os.path.exists(clip_dir_path):
                os.makedirs(clip_dir_path)

            clip_start_time = random.choice(range(int(video.duration - parameters['clip_length_seconds'])))
            clip_time = (clip_start_time, clip_start_time + parameters['clip_length_seconds'])
            clip_fname = f'{clip_time[0]}_{clip_time[1]}__{name}'
            clip_fpath = os.path.join(clip_dir_path, clip_fname)

            subclip = video.subclip(*clip_time)
            subclip = resize(subclip, width=parameters['clip_width'])
            try:
                subclip.write_videofile(clip_fpath, fps=parameters['clip_fps'])
            finally:
                subclip.close()
        finally:
            video.close()
