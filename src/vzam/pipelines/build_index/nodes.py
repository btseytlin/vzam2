import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from kedro.framework.session import get_current_session
from vzam.indexer import Indexer


def build_index(video_order2_features, parameters):
    indexer = Indexer(**parameters['indexer_params'])

    features = []
    ids = []

    for name, feature_vector in tqdm(video_order2_features.items()):
        features.append(feature_vector)
        ids.append(name)

    features = np.array(features).astype(np.float32)
    ids = np.array(ids)

    indexer.train(features)
    indexer.add(features, ids)
    assert indexer.index.ntotal == len(features) == len(indexer.metadata) == len(ids)

    return indexer
