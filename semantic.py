import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """
    Load a SentenceTransformer embedding model from HF.
    """
    return SentenceTransformer(model_name)


def compute_embeddings(texts, model, batch_size=EMBEDDING_BATCH_SIZE):
    """
    Encode a list of texts into a 2D numpy array.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def normalize_embeddings(embeddings):
    """
    L2-normalize embeddings.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # avoid dividing by zero by clipping small norms 
    norms = np.clip(norms, 1e-9, None)
    return embeddings / norms


def build_faiss_index(embeddings):
    """
    Build a FAISS index
    """
    dim = embeddings.shape[1]
    # brute force inner product index 
    index = faiss.IndexFlatIP(dim)
    # add all embeddings to the index 
    index.add(embeddings)
    return index


def semantic_search(query, model, index, k):
    """
    Given a query string, return top-k indices in the FAISS index.
    """
    # encode query into embedding
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    query_emb = normalize_embeddings(query_emb)
    scores, indices = index.search(query_emb, k)
    # indices has shape (1, k) so we take only the first row
    return indices[0].tolist()