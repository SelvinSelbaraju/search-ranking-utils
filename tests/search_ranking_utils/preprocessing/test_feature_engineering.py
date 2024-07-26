import time
import logging
import pytest
from search_ranking_utils.preprocessing.feature_engineering import (
    calculate_text_cosine_similarity,
    get_timestamp_part,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def test_calculate_text_cosine_similarity(dummy_df):
    scores = calculate_text_cosine_similarity(
        dummy_df, "search_query", "product_title"
    )
    logging.info(scores)
    assert scores.shape == (len(dummy_df),)
    for score in scores:
        assert score > -1 and score < 1


def test_calculate_text_cosine_similarity_inference_time(dummy_df):
    NUM_INFERENCES = 15000
    THRESHOLD_SECONDS = 60
    inference_df = dummy_df[["search_query", "product_id"]].sample(
        n=NUM_INFERENCES, replace=True
    )
    start = time.time()
    calculate_text_cosine_similarity(
        inference_df, "search_query", "product_id"
    )
    end = time.time()
    elapsed = end - start
    logging.info(f"Elapsed Time: {elapsed}")
    assert elapsed <= THRESHOLD_SECONDS


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