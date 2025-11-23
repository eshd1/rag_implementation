import nltk
from nltk.tokenize import sent_tokenize
from config import SENTENCES_PER_CHUNK

def ensure_nltk():
    try:
        _ = sent_tokenize("Test.")
    except LookupError:
        nltk.download("punkt")


def chunk_documents(docs, sentences_per_chunk=SENTENCES_PER_CHUNK):
    """
    Convert documents to sentence-based chunks.
    """
    # make sure NLTK is ready 
    ensure_nltk()

    chunks = []
    chunk_doc_ids = []
    doc_to_chunks = {}

    # go through each of the docs 
    for doc_id, text in enumerate(docs):
        # split the document into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            # skip any empty docs 
            continue
        
        for i in range(0, len(sentences), sentences_per_chunk):
            # join a window of sentences into a chunk 
            chunk = " ".join(sentences[i : i + sentences_per_chunk]).strip()
            if not chunk:
                continue
            
            # assign a new chunk id 
            chunk_id = len(chunks)
            chunks.append(chunk)
            chunk_doc_ids.append(doc_id)

            # track mapping doc to its chunks 
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = []
            doc_to_chunks[doc_id].append(chunk_id)

    return chunks, chunk_doc_ids, doc_to_chunks