import numpy as np
import faiss


class Indexer:
    def __init__(self, nlist, d, M, nbits, hnsw_m, quantizer=None, index=None, metadata=None):
        self.nlist = nlist
        self.d = d
        self.M = M
        self.nbits = nbits
        self.hnsw_m = hnsw_m

        self.quantizer = quantizer or faiss.IndexHNSWFlat(self.d, self.hnsw_m)
        self.index = index or faiss.IndexIVFPQ(self.quantizer, self.d, self.nlist, self.M, self.nbits)
        self.metadata = metadata or {}

    @property
    def is_trained(self):
        return self.index.is_trained

    def add_meta(self, meta):
        assert self.index.ntotal == len(self.metadata) + len(meta)
        index_ids = range(self.index.ntotal-len(meta), self.index.ntotal)
        for index_id, m in zip(index_ids, meta):
            self.metadata[index_id] = m

    def train(self, vectors, meta=None):
        self.index.train(vectors)
        if meta:
            self.add_meta(meta)

    def add(self, vectors, meta):
        self.index.add(vectors)
        self.add_meta(meta)

    def get_match_indices(self, query_features):
        distances, indices = self.index.search(query_features, k=self.nlist)

        match_indices = [[j for j in indices[i] if j != -1] for i in range(len(indices))]
        return indices, distances, match_indices

    def query(self, vectors, times):
        indices, distances, match_indices = self.get_match_indices(vectors)

        flat_idx = indices.flatten()
        flat_idx = flat_idx[flat_idx != -1]
        match_candidate_labels = list(set([self.metadata[m]['label'] for m in flat_idx]))

        match_time_pairs = {}
        for label in match_candidate_labels:
            match_time_pairs[label] = []
            # get match pairs for label
            for i, matches in enumerate(indices):
                for match_peak_idx in matches:
                    if match_peak_idx == -1:
                        continue
                    match_time, match_label = self.metadata[match_peak_idx]['time'], self.metadata[match_peak_idx]['label']
                    if match_label != label:
                        continue
                    match_time_pairs[label].append((times[i], match_time))
            match_time_pairs[label] = np.array(match_time_pairs[label])

        candidate_scores = {}
        for candidate in match_time_pairs.keys():
            time_pairs = match_time_pairs[candidate]
            query_times = time_pairs[:, 0]
            match_times = time_pairs[:, 1]
            hist, bin_edges = np.histogram(query_times - match_times, bins=10)
            candidate_scores[candidate] = max(hist)

            # plt.title(f'{candidate} match scatterplot')
            # plot_match_scatterplot(query_times, match_times)
            # plt.title(f'{candidate} match hist')
            # plot_match_histogram(query_times, match_times)

        return candidate_scores

