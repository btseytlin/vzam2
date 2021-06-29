from kedro.pipeline import Pipeline, node

from .nodes import build_index


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                build_index,
                ["video_keyframes", "parameters"],
                "indexer"
            ),
        ]
    )
