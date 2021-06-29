from kedro.pipeline import Pipeline, node

from .nodes import get_video_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_video_features,
                ["train_videos", "parameters"],
                None #"video_features"
            ),
        ]
    )
