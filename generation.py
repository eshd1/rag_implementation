import os
import requests

from config import (
    HF_INFERENCE_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)


def build_prompt(query, chunk_texts):
    """
    Construct a RAG-style prompt 
    """
    sources_str_parts = []
    for i, chunk in enumerate(chunk_texts, start=1):
        sources_str_parts.append(f"[{i}] {chunk.strip()}")

    # join sources with blank lines for readbility 
    sources_str = "\n\n".join(sources_str_parts)

    # put everything together into instructions 
    prompt = (
        "You are a helpful assistant. You must answer using ONLY the information "
        "provided in the numbered sources below.\n\n"
        f"User query:\n{query}\n\n"
        "Sources:\n"
        f"{sources_str}\n\n"
        "Instructions:\n"
        "- Use the sources to answer the query.\n"
        "- If you are unsure or the sources do not contain the answer, say so explicitly.\n"
        "- Do not invent facts beyond the sources.\n\n"
        "Answer:"
    )
    return prompt


def hf_generate(
    prompt,
    model_id=HF_INFERENCE_MODEL,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    """
    Call HuggingFace Inference API for text generation.
    Requires HF_API_TOKEN in environment.
    """
    api_token = os.environ.get("HF_API_TOKEN")
    if api_token is None:
        raise RuntimeError("HF_API_TOKEN env var is not set.")
    
    # HF inference endpoint for this model
    url = f"https://api-inference.huggingface.co/models/{model_id}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict) and "generated_text" in item:
            return str(item["generated_text"])
        return str(item)

    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"])

    # stringify whatever we get
    return str(data)