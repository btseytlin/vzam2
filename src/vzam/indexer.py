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
        # self.index = index or faiss.IndexIVFPQ(self.quantizer, self.d, self.nlist, self.M, self.nbits)
        self.index = index or faiss.IndexFlatIP(self.d)
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
        faiss.normalize_L2(vectors)
        self.index.train(vectors)
        if meta:
            self.add_meta(meta)

    def add(self, vectors, meta):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.add_meta(meta)

    def get_match_indices(self, query_features):
        distances, indices = self.index.search(query_features, k=self.nlist)

        match_indices = [[j for j in indices[i] if j != -1] for i in range(len(indices))]
        return indices, distances, match_indices

    def query(self, vectors):
        faiss.normalize_L2(vectors)
        distances, indices = self.index.search(vectors, k=self.nlist)



        flat_idx = indices.flatten()
        match_mask = flat_idx != -1
        flat_idx = flat_idx[match_mask]
        flat_distances = distances.flatten()
        flat_distances = flat_distances[match_mask]

        candidate_scores = {self.metadata[m]: d for m, d in zip(flat_idx, flat_distances)}

        return candidate_scores

