from config import (
    BM25_B,
    BM25_K1,
    TOP_K,
    BBC_DATASET_NAME,
    EMBEDDING_MODEL_NAME,
)
from data import load_bbc_dataset
from chunking import chunk_documents
from bm25 import BM25
from semantic import (
    load_embedding_model,
    compute_embeddings,
    normalize_embeddings,
    build_faiss_index,
)
from retrieval import BM25Retriever, SemanticRetriever
from generation import build_prompt, hf_generate
from evaluation import evaluate_retrieval_at_k, compute_rouge


def build_retrievers(chunks):
    """Build BM25 and semantic retrievers."""
    # BM25
    bm25 = BM25(k1=BM25_K1, b=BM25_B)
    bm25.fit(chunks) # index all chunks 
    bm25_retriever = BM25Retriever(bm25)

    # Semantic
    emb_model = load_embedding_model(EMBEDDING_MODEL_NAME) # load embeddings for all chunks
    raw_embs = compute_embeddings(chunks, emb_model)
    # normalize to unit length
    norm_embs = normalize_embeddings(raw_embs)
    index = build_faiss_index(norm_embs)
    semantic_retriever = SemanticRetriever(emb_model, index)

    return bm25_retriever, semantic_retriever


def evaluate_retrievers(
    summaries,
    bm25_retriever,
    semantic_retriever,
    doc_to_chunks,
    k=TOP_K,
    n_eval=200,
):
    """
    Compare BM25 and semantic retrieval with recall@k / precision@k.
    Uses summary text as query; relevant chunks are from the same document.
    """
    bm25_results = {}
    sem_results = {}

    # limit n_eval to dataset size 
    n_eval = min(n_eval, len(summaries))

    # for each doc use its summary as a query 
    for doc_id in range(n_eval):
        query = summaries[doc_id]

        # retrieve top-k chunk ids from each retriever 
        bm25_results[doc_id] = bm25_retriever.retrieve(query, k)
        sem_results[doc_id] = semantic_retriever.retrieve(query, k)

    # comoute metrics for BM25 and semantic 
    bm25_scores = evaluate_retrieval_at_k(bm25_results, doc_to_chunks, k)
    sem_scores = evaluate_retrieval_at_k(sem_results, doc_to_chunks, k)

    print(
        f"BM25 precision@{k}: {bm25_scores['precision_at_k']:.4f}, "
        f"recall@{k}: {bm25_scores['recall_at_k']:.4f}"
    )
    print(
        f"SEMANT precision@{k}: {sem_scores['precision_at_k']:.4f}, "
        f"recall@{k}: {sem_scores['recall_at_k']:.4f}"
    )

    return bm25_scores, sem_scores


def evaluate_generation_with_retriever(texts, summaries, chunks, retriever, k=TOP_K, n_eval=50, label="BM25"):
    """
    RAG-style generation evaluation with ROUGE.
    """
    predictions = []
    references = []

    n_eval = min(n_eval, len(texts), len(summaries))

    for doc_id in range(n_eval):
        query = summaries[doc_id]
        top_chunk_ids = retriever.retrieve(query, k)
        source_chunks = [chunks[cid] for cid in top_chunk_ids]

        # this builds the rag style prompt 
        prompt = build_prompt(query, source_chunks)
        answer = hf_generate(prompt) # call HF inference API 
        predictions.append(answer)
        references.append(summaries[doc_id])

    # compute rogue metrics 
    rouge_scores = compute_rouge(predictions, references)
    print(f"{label} RAG ROUGE scores:")
    for key, val in rouge_scores.items():
        print(f"{key}: {val:.4f}")

    return rouge_scores


def main():
    # Load dataset
    print("Loading BBC dataset...")
    texts, summaries = load_bbc_dataset(BBC_DATASET_NAME)

    # Chunk documents
    print("Chunking documents into sentence-based chunks...")
    chunks, chunk_doc_ids, doc_to_chunks = chunk_documents(texts)

    print(f"Total docs: {len(texts)}")
    print(f"Total chunks: {len(chunks)}")

    # Build retrievers
    print("Building BM25 and semantic retrievers...")
    bm25_retriever, semantic_retriever = build_retrievers(chunks)

    # Evaluate retrieval
    print("Evaluating retrieval (precision@5 / recall@5)...")
    evaluate_retrievers(summaries=summaries, bm25_retriever=bm25_retriever, semantic_retriever=semantic_retriever, doc_to_chunks=doc_to_chunks, k=TOP_K, n_eval=200)

    # Evaluate RAG + generation (ROUGE) 
    # this will call the HF API 
    print("Evaluating generation with BM25-based RAG (ROUGE)")
    evaluate_generation_with_retriever(texts=texts, summaries=summaries, chunks=chunks, retriever=bm25_retriever, k=TOP_K, n_eval=20, label="BM25")

    print("Evaluating generation with semantic-based RAG (ROUGE)")
    evaluate_generation_with_retriever(texts=texts, summaries=summaries, chunks=chunks, retriever=semantic_retriever, k=TOP_K, n_eval=20, label="Semantic")


if __name__ == "__main__":
    main()