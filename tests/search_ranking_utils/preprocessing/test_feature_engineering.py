import pytest
from search_ranking_utils.preprocessing.feature_engineering import (
    calculate_text_cosine_similarity,
    get_timestamp_part,
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


@pytest.mark.parametrize(
    argnames=("timestamp_str", "timestamp_part", "expected"),
    argvalues=(
        ["2024-07-01 01:00:00", "hour", 1],
        ["1999-12-06 12:56:45", "second", 45],
        ["2024-07-26 12:56:45", "weekday", 4],
    ),
)
def test_get_timestamp_part(timestamp_str, timestamp_part, expected):
    result = get_timestamp_part(timestamp_str, timestamp_part)
    assert result == expected
