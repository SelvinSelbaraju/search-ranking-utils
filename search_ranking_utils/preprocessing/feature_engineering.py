from sentence_transformers import SentenceTransformer, util

EMBEDDING_MODEL = "msmarco-MiniLM-L-6-v3"
model = SentenceTransformer(EMBEDDING_MODEL)


def calculate_text_cosine_similarity(text_1: str, text_2: str) -> float:
    """
    Given two pieces of text, calculate their similarity score
    Use the sentence_transformers to embed the pieces of text
    Then take the cosine similarity
    """
    embedding_1 = model.encode(text_1)
    embedding_2 = model.encode(text_2)
    return util.cos_sim(embedding_1, embedding_2)
