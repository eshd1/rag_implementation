import statistics
import evaluate


def evaluate_retrieval_at_k(results, doc_to_chunks, k):
    """
    Compute mean precision@k and recall@k.
    """
    precisions = []
    recalls = []

    for doc_id, retrieved_chunk_ids in results.items():
        relevant_chunks = set(doc_to_chunks.get(doc_id, []))
        if not relevant_chunks:
            # fallback if no chunks, but this shouldnt happen
            continue
        
        # take the top-k retrieved chunks
        top_k = retrieved_chunk_ids[:k]
        # count how many are relevant 
        hits = sum(1 for cid in top_k if cid in relevant_chunks)

        # calculate precision and recall 
        precisions.append(hits / float(k))
        recalls.append(hits / float(len(relevant_chunks)))

    return {
        "precision_at_k": statistics.fmean(precisions) if precisions else 0.0,
        "recall_at_k": statistics.fmean(recalls) if recalls else 0.0,
    }


def compute_rouge(predictions, references):
    """
    Compute ROUGE scores for predictions vs references using HF's evaluate.
    """
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    return scores