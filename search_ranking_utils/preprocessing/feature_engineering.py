from datetime import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "msmarco-MiniLM-L-6-v3"
model = SentenceTransformer(EMBEDDING_MODEL)


def calculate_text_cosine_similarity(
    df: pd.DataFrame, text_col_1: str, text_col_2: str
) -> np.ndarray:
    """
    Given two pieces of text, calculate their similarity score
    Use the sentence_transformers to embed the pieces of text
    Then take the cosine similarity

    Batching the data makes this much quicker
    """
    text_embeddings_1 = model.encode(df[text_col_1].values)
    text_embeddings_2 = model.encode(df[text_col_2].values)
    # We only care about the diagonal as that holds the right similarity scores
    # Eg. [0][0] is first search query with first product
    # [4][4] is fifth search query with fifth product
    # We would not ever want i != j
    return np.diag(model.similarity(text_embeddings_1, text_embeddings_2))


def get_timestamp_part(
    timestamp_str: str,
    timestamp_part: str,
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
) -> int:
    """
    Given a timestamp and the format, get a part of it
    Eg. Get what the hour is or the day of the week
    Monday is 0 indexed
    """
    dt = datetime.strptime(timestamp_str, timestamp_format)
    # Weekday is a method not an attribute
    if timestamp_part == "weekday":
        return dt.weekday()
    else:
        return getattr(dt, timestamp_part)
