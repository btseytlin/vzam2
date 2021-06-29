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

    def train(self, vectors):
        return self.index.train(vectors)

    def add(self, vectors, meta):
        self.index.add(vectors)

        index_ids = range(self.index.ntotal, self.index.ntotal + len(vectors))
        for index_id, m in zip(index_ids, meta):
            self.metadata[index_id] = m

