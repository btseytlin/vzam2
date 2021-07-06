from kedro.pipeline import Pipeline, node

from .nodes import obtain_cv_splits, run_cv


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                obtain_cv_splits,
                ["video_features", "clip_features", "parameters"],
                "cv_splits"
            ),
            node(
                run_cv,
                ["cv_splits", "video_features", "clip_features", "parameters"],
                "evaluation_metrics"
            ),
        ]
    )
