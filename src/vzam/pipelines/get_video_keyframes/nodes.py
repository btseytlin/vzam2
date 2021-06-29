import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import ruptures as rpt


def reduce_to_series(features):
    return PCA(n_components=1).fit_transform(features).flatten()


def get_peaks(series):
    algo = rpt.KernelCPD(kernel="rbf", min_size=10).fit(series)
    result = algo.predict(pen=2)[:-1]
    if len(result) < 5:
        result = range(0, len(series), len(series)//5)
    else:
        new_result = []
        for i in range(1, len(result)):
            left = result[i-1]
            right = result[i]
            new_result.append(left)
            new_result.append((left+right)//2)
            new_result.append(right)
        result = np.array(new_result)
    return result


def get_video_keyframes(video_features, parameters):
    keyframes = {}
    for dir_entry in video_features:
        path = dir_entry.path

        df = pd.read_csv(path)
        df.index = df.time
        times = df.time.values
        df = df.drop(columns=['time'])
        features = df.values

        series = reduce_to_series(features)
        keyframe_indices = get_peaks(series)
        keyframe_times = times[keyframe_indices]
        keyframes[path] = {'indices': keyframe_indices.astype(int).tolist(),
                           'times': keyframe_times.round(3).astype(float).tolist()}

    return keyframes