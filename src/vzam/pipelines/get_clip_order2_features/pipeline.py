from kedro.pipeline import Pipeline, node
from .nodes import get_clip_order2_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_clip_order2_features,
                ["clip_features", "order2_extractor", "parameters"],
                None#"clip_order2_features"
            ),
        ]
    )
