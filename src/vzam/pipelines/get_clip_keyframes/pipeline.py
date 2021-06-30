from kedro.pipeline import Pipeline, node

from ..get_video_keyframes.nodes import get_video_keyframes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_video_keyframes,
                ["clip_features", "parameters"],
                "clip_keyframes"
            ),
        ]
    )
