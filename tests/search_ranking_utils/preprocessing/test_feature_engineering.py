import pytest
from search_ranking_utils.preprocessing.feature_engineering import (
    calculate_text_cosine_similarity,
)


@pytest.mark.parametrize(
    argnames=["text_1", "text_2"],
    argvalues=[
        ("water bottle", "bottle of water"),
        ("flat screen tv", "a relaxing otter"),
    ],
)
def test_calculate_text_cosine_similarity(text_1, text_2):
    score = calculate_text_cosine_similarity(text_1, text_2)
    assert score > -1 and score < 1
