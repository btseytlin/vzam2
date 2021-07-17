from kedro.pipeline import Pipeline, node

from .nodes import get_video_order2_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_video_order2_features,
                ["video_features", "order2_extractor", "parameters"],
                None, #"video_order2_features"
            ),
        ]
    )
