import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from ..build_index.nodes import build_index
from ..train_order2_extractor.nodes import train_order2_extractor
from ..get_video_order2_features.nodes import get_video_order2_features


def obtain_cv_splits(video_features, clip_features, parameters):
    video_names = np.array([e.name for e in video_features])
    clip_names = np.array([e.name for e in clip_features])

    cv = KFold(n_splits=parameters['n_splits'])

    splits = []
    for train_index, test_index in cv.split(video_names):
        train_videos = video_names[train_index]
        clip_source_name = [clip.split('__')[1] for clip in clip_names]
        clip_labels = {clip_name: (src_name if src_name in train_videos else 'missing') for (clip_name, src_name) in zip(clip_names, clip_source_name)}
        split = {
            "train_videos": train_videos.tolist(),
            "train_index": train_index.tolist(),
            "clip_names": clip_names.tolist(),
            "clip_labels": clip_labels,
        }
        splits.append(split)

    return splits


def run_cv(cv_splits, video_features, clip_features, parameters):
    video_features = list(video_features)
    clip_features = list(clip_features)
    split_metrics = []

    for i, split in tqdm(enumerate(cv_splits), total=len(cv_splits)):
        train_videos = split['train_videos']
        clip_labels = split['clip_labels']
        clip_names = split['clip_names']

        train_video_features = [entry for entry in video_features if entry.name in train_videos]
        train_clip_features = [entry for entry in clip_features if entry.name.split('__')[1] in train_videos]

        order2_extractor = train_order2_extractor(train_video_features, train_clip_features, parameters)

        train_video_order2_features = get_video_order2_features(train_video_features, order2_extractor, parameters)
        train_clip_order2_features = get_video_order2_features(train_clip_features, order2_extractor, parameters)

        indexer = build_index(train_video_order2_features, parameters)

        true_labels = []
        pred_labels = []
        pred_scores = []
        for clip_name, clip_order2_feature_vector in tqdm(train_clip_order2_features.items(), total=len(train_clip_order2_features)):

            batch = np.expand_dims(np.array(clip_order2_feature_vector).astype(np.float32), axis=0)
            scores = indexer.query(batch)
            scores = {k: round(float(v), 4) for k, v in scores.items()}
            pred_scores.append(scores)

            correct_label = clip_labels[clip_name]
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

