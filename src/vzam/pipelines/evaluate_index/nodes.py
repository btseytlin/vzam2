import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from ..build_index.nodes import build_index, train_indexer
from ..train_order2_extractor.nodes import train_order2_extractor
from ..get_video_order2_features.nodes import  extract_order2_features
from ...indexer import Indexer
from ...utils import fname


def obtain_cv_splits(video_features, clip_features, parameters):
    video_names = np.array([fname(e.name) for e in video_features])
    clip_names = np.array([fname(e.name)  for e in clip_features])

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

        train_video_features = [e for e in video_features if fname(e.name) in train_videos]
        train_clip_features = [e for e in clip_features if fname(e.name).split('__')[1] in train_videos]

        order2_extractor = train_order2_extractor(train_video_features, parameters)

        train_video_source_names, train_video_clip_times, train_video_order2_features = extract_order2_features(train_video_features, order2_extractor, parameters)
        train_clip_source_names, train_clip_clip_times, train_clip_order2_features = extract_order2_features(train_clip_features, order2_extractor, parameters)

        indexer = train_indexer(Indexer(**parameters['indexer_params']),
                                train_video_order2_features,
                                train_video_source_names,
                                train_video_clip_times)

        group_clip_vectors = {name: [] for name in train_clip_source_names}
        for name, feature_vector, clip_time in zip(train_clip_source_names, train_clip_order2_features, train_clip_clip_times):
            group_clip_vectors[name].append((feature_vector, clip_time))

        for k in group_clip_vectors:
            group_clip_vectors[k] = sorted(group_clip_vectors[k], key=lambda x: x[1][0])
        true_labels = []
        pred_labels = []
        pred_scores = []

        for clip_name in tqdm(group_clip_vectors):
            clip_feature_vectors = [t[0] for t in group_clip_vectors[clip_name]]
            clip_times = [t[1] for t in group_clip_vectors[clip_name]]
            vectors_batch = np.stack(clip_feature_vectors).astype(np.float32)
            scores = indexer.query(vectors_batch, clip_times)

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

