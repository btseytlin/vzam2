from kedro.pipeline import Pipeline, node

from .nodes import cut_videos_into_clips


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cut_videos_into_clips,
                ["train_videos", "parameters"],
                "video_clips"
            ),
        ]
    )
