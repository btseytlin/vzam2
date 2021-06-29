from kedro.pipeline import Pipeline, node

from .nodes import get_video_keyframes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_video_keyframes,
                ["video_features", "parameters"],
                "video_keyframes"
            ),
        ]
    )
