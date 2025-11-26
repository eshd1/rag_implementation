# config.py

BBC_DATASET_NAME = "gopalkalpande/bbc-news-summary"
TEXT_FIELD = "Articles" 
SUMMARY_FIELD = "Summaries"

# chunking
SENTENCES_PER_CHUNK = 5

# settings for BM25
BM25_K1 = 1.5
BM25_B = 0.75

# embedding and Faiss settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
TOP_K = 5

# HF Inference API w/ openai client (see generation.py)
HF_INFERENCE_MODEL = "HuggingFaceTB/SmolLM3-3B:hf-inference"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.95