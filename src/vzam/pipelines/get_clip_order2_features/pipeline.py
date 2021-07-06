from kedro.pipeline import Pipeline, node
from vzam.pipelines.get_video_order2_features.nodes import get_video_order2_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_video_order2_features,
                ["clip_features", "order2_extractor", "parameters"],
                "clip_order2_features"
            ),
        ]
    )
