from kedro.pipeline import Pipeline, node

from .nodes import get_clip_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_clip_features,
                ["video_clips", "parameters"],
                None # video_features
            ),
        ]
    )
