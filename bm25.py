import math
from text_utils import simple_tokenize


class BM25:
    """
    BM25 implementation over a list of documents.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.doc_len = []
        self.inverted_index = {} # inverted index 
        # idf values
        self.idf = {}

    def fit(self, documents):
        """
        Build inverted index and IDF stats from all of the documents.
        """
        self.N = len(documents)
        self.doc_len = []
        # document frequency
        df = {}
        # build the inverted index and compute doc frequencies. 
        for doc_id, doc in enumerate(documents):
            terms = simple_tokenize(doc)
            self.doc_len.append(len(terms))

            # track what we have already seen 
            seen_terms = set()
            for term in terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                # initialize doc entry if needed 
                if doc_id not in self.inverted_index[term]:
                    self.inverted_index[term][doc_id] = 0
                self.inverted_index[term][doc_id] += 1

                if term not in seen_terms:
                    df[term] = df.get(term, 0) + 1
                    seen_terms.add(term)

        self.avgdl = sum(self.doc_len) / max(1, self.N)
        
        # compute idf for each term using bm25 formula 
        for term, freq in df.items():
            # Standard BM25 idf
            self.idf[term] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def _score_term(self, term, doc_id, freq):
        """
        BM25 contribution of a single term for a doc.
        """
        idf = self.idf.get(term, 0.0)
        dl = self.doc_len[doc_id]
        # bm 25 denominator factor 
        denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(1.0, self.avgdl))
        return idf * (freq * (self.k1 + 1.0) / max(denom, 1e-9)) # avoid division by zero

    def score(self, query):
        """
        Score all documents  for a query.
        """
        query_terms = simple_tokenize(query)
        scores = {}

        for term in query_terms:
            if term not in self.inverted_index:
                # if the term doesnt appear anywhere skip it 
                continue

            postings = self.inverted_index[term]
            for doc_id, freq in postings.items():
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += self._score_term(term, doc_id, freq)
        # turn dict into a list and sort by score dscription 
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def top_k(self, query, k):
        """
        Return top-k document indices for a query.
        """
        ranked = self.score(query)
        return [doc_id for doc_id, _ in ranked[:k]]