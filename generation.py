# generation.py

import os
from openai import OpenAI  

from config import (
    HF_INFERENCE_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)


def build_prompt(query, chunk_texts):
    sources_str_parts = []
    for i, chunk in enumerate(chunk_texts, start=1):
        sources_str_parts.append(f"[{i}] {chunk.strip()}")
    sources_str = "\n\n".join(sources_str_parts)

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
    Call Hugging Face Router using the OpenAI-compatible chat API.
    """
    api_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
    if api_token is None:
        raise RuntimeError("HF_TOKEN or HF_API_TOKEN env var must be set.")

    # openAI client pointing at HF Router
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_token,
    )

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # get the text from the first choice
    return completion.choices[0].message.content