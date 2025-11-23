from datasets import load_dataset
from config import BBC_DATASET_NAME, TEXT_FIELD, SUMMARY_FIELD


def load_bbc_dataset(dataset_name=BBC_DATASET_NAME, split=None, max_samples=None):
    """
    Load the BBC dataset from HuggingFace and return lists of texts and summaries.
    """
    dataset_dict = load_dataset(dataset_name)

    # if split is not picked take the first one 
    if split is None:
        split = list(dataset_dict.keys())[0]

    ds = dataset_dict[split]

    # can limit number of samples 
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = []
    summaries = []
    
    # loop through rows and collect text and summary. 
    for row in ds:
        texts.append(str(row[TEXT_FIELD]))
        summaries.append(str(row[SUMMARY_FIELD]))

    return texts, summaries