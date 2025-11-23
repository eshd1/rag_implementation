import re


def simple_tokenize(text):
    """
    simple word tokenizer for BM25.
    """
    # \w+ matches sequences of word chaacters
    return re.findall(r"\w+", text.lower())