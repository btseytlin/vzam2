from kedro.pipeline import Pipeline, node

from .nodes import train_order2_extractor


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_order2_extractor,
                ["video_features", "parameters"],
                "order2_extractor",
            ),
        ]
    )