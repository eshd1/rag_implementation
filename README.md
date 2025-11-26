# BBC News RAG – BM25 vs Semantic Retrieval

A small RAG pipeline on the BBC news summary dataset.  
Compares **BM25 (lexical)** vs **semantic retrieval (embeddings + FAISS)**, then sends the top-k chunks to an LLM via the HuggingFace Inference API.

---

## Models Used

- **Dataset:** `gopalkalpande/bbc-news-summary`
- **Lexical retrieval:** from scratch BM25 implementation over sentence-based chunks
- **Semantic retrieval:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM (generation via HF Inference API):** `HuggingFaceTB/SmolLM3-3B:hf-inference`  
  (configured in `config.py` as `HF_INFERENCE_MODEL`)

---

## Repository Structure

```text
.
├─ config.py          # Config file
├─ requirements.txt   # Python dependencies
├─ data.py            # Load BBC dataset from HuggingFace
├─ chunking.py        # Sentence-based document chunking
├─ text_utils.py      # Simple tokenizer for BM25
├─ bm25.py            # BM25 implementation from scratch 
├─ semantic.py        # Embeddings + FAISS index and search
├─ retrieval.py       # BM25Retriever / SemanticRetriever
├─ generation.py      # Build prompt, call HF Inference API
├─ evaluation.py      # Precision@k, Recall@k, ROUGE
└─ run_experiment.py  # main file to run
```
## Steps to Setup and Run Project

- ```git clone https://github.com/eshd1/rag_implementation.git```
- ```cd rag-implementation```
- ```python -m venv rag```
- ```pip install -r requirements.txt```
- ```export HF_API_TOKEN="your_hf_inference_api_token```
- Run ```python run_experiment.py``` to run the experiement 