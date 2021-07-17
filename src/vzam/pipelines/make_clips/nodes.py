import os
import random
from tqdm.auto import tqdm
from kedro.framework.session import get_current_session
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.resize import resize


def cut_videos_into_clips(train_videos, parameters):
    session = get_current_session()
    context = session.load_context()
    out_dataset_path = context.catalog.datasets.video_clips._dir_path
    if not os.path.exists(out_dataset_path):
        os.makedirs(out_dataset_path)

    for video_dir_enty in tqdm(train_videos):
        fpath = video_dir_enty.path
        name = video_dir_enty.name
        basename = '.'.join(name.split('.')[:-1])

        clip_dir_path = os.path.join(out_dataset_path, basename)
        if not os.path.exists(clip_dir_path):
            os.makedirs(clip_dir_path)

        video = VideoFileClip(fpath)
        total_duration = int(video.duration)
        try:
            for clip_start in range(0, total_duration, total_duration//parameters['clips_per_video']):
                if clip_start + parameters['clip_length_seconds'] > total_duration:
                    break
                clip_time = (clip_start, clip_start + parameters['clip_length_seconds'])
                subclip = video.subclip(*clip_time)
                subclip = resize(subclip, width=parameters['clip_width'])

                clip_fname = f'{clip_time[0]}_{clip_time[1]}__{name}'
                clip_fpath = os.path.join(clip_dir_path, clip_fname)
                subclip.write_videofile(clip_fpath, fps=parameters['clip_fps'])
        finally:
            video.close()
