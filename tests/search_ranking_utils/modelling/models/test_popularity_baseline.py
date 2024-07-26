import pytest
import pandas as pd
from search_ranking_utils.modelling.models.popularity_baseline import (
    PopularityBaseline,
)


def test_populatiy_baseline_fit(dummy_df, dummy_schema):
    model = PopularityBaseline(
        dummy_schema["target"],
        "product_id",
    )
    model.fit(dummy_df)

    # Check a score lookup exists
    assert model.score_lookup


@pytest.mark.parametrize(
    argnames=["default_val", "expected_default_val"],
    argvalues=[
        (0.5, 0.5),
        (None, 2 / 7),
    ],
)
def test_populatiy_baseline_call(
    dummy_df, dummy_schema, default_val, expected_default_val
):
    model = PopularityBaseline(
        dummy_schema["target"], "product_id", default_val=default_val
    )
    model.fit(dummy_df)
    scores = model(dummy_df)

    # Check the scores
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[4] == 1.0

    # Check the OOV products
    # Add on a row of data and change the product_id
    df = pd.concat([dummy_df, dummy_df.iloc[0]])
    df.iloc[-1]["product_id"] = "prod200"
    assert model(df)[-1] == expected_default_val
