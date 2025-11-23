from bm25 import BM25
from semantic import semantic_search
from config import TOP_K

# two warpper classes around BM25 and semantic search
class BM25Retriever:
    def __init__(self, bm25):
        self.bm25 = bm25

    def retrieve(self, query, k=TOP_K):
        return self.bm25.top_k(query, k)


class SemanticRetriever:
    def __init__(self, model, index):
        self.model = model
        self.index = index

    def retrieve(self, query, k=TOP_K):
        return semantic_search(query, self.model, self.index, k)