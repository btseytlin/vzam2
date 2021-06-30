import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from ..build_index.nodes import build_index


def obtain_cv_splits(video_features, clip_features, parameters):
    video_feature_files = np.array([entry.name for entry in video_features])
    clip_feature_files = np.array(sorted([entry.name for entry in clip_features]))

    cv = KFold(n_splits=parameters['n_splits'])

    splits = []
    for train_index, test_index in cv.split(video_feature_files):
        train_videos = video_feature_files[train_index]
        clip_source_name = [clip.split('__')[1] for clip in clip_feature_files]
        clip_labels = [(src_name if src_name in train_videos else 'missing') for src_name in clip_source_name]
        split = {
            "train_videos": train_videos.tolist(),
            "train_index": train_index.tolist(),
            "clip_features_fnames": clip_feature_files.tolist(),
            "clip_labels": clip_labels,
        }
        splits.append(split)

    return splits


def run_cv(cv_splits, video_features, video_keyframes, clip_features, clip_keyframes, parameters):
    video_features = list(video_features)
    split_metrics = []

    clip_features_name_to_path = {entry.name: entry.path for entry in clip_features}

    clip_keyframe_features = {}
    clip_keyframe_ids = {}
    for name in clip_keyframes:
        features_path = clip_features_name_to_path[name]
        df = pd.read_csv(features_path)
        df.index = df.time
        df = df.drop(columns=['time'])
        features = df.values.astype(np.float32)

        keyframe_indices = clip_keyframes[name]['indices']
        keyframe_times = clip_keyframes[name]['times']
        keyframe_features = features[keyframe_indices]
        keyframe_ids = [dict(time=t, label=name) for t in keyframe_times]

        clip_keyframe_features[name] = keyframe_features
        clip_keyframe_ids[name] = keyframe_ids

    for i, split in tqdm(enumerate(cv_splits), total=len(cv_splits)):
        train_videos = split['train_videos']
        clip_labels = split['clip_labels']
        clip_features_fnames = split['clip_features_fnames']

        train_video_keyframes = {k: v for k, v in video_keyframes.items() if k in train_videos}
        train_video_features = [entry for entry in video_features if entry.name in train_videos]

        indexer = build_index(train_video_keyframes, train_video_features, parameters)

        true_labels = []
        pred_labels = []
        pred_scores = []
        for j, clip_fname in tqdm(enumerate(clip_features_fnames), total=len(clip_features_fnames)):
            query_keyframe_features = clip_keyframe_features[clip_fname]
            query_ids = clip_keyframe_ids[clip_fname]
            query_timescodes = [query_id['time'] for query_id in query_ids]

            scores = indexer.query(query_keyframe_features, query_timescodes)
            scores = {k: float(v) for k, v in scores.items()}
            pred_scores.append(scores)

            correct_label = clip_labels[j]
            true_labels.append(correct_label)

            predicted_label = 'missing'
            if scores:
                max_score_label = max(scores, key=scores.get)
                if scores[max_score_label] > parameters['threshold']:
                    predicted_label = max_score_label

            pred_labels.append(predicted_label)

        # Compute metrics

        metrics = {
            'split_num': i,
            'classification_report': classification_report(true_labels, pred_labels),
            'pred_labels': pred_labels,
            'true_labels': true_labels,
            'pred_scores': pred_scores,
        }
        split_metrics.append(metrics)

    return split_metrics

