import pytest
import numpy as np
import pandas as pd
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset
from search_ranking_utils.modelling.model_factory import ModelFactory
from search_ranking_utils.evaluation.feature_importance import (
    calculate_feature_importance,
    _generate_background,
)


@pytest.fixture
def dummy_model(dummy_trainable_df, dummy_schema):

    model = ModelFactory.get_instance_from_config(
        "sklearn.linear_model:LogisticRegression"
    )
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    model.fit(X, y)
    return model


def test_generate_background():
    df = pd.DataFrame(
        {"num1": [1.0, 0.0, 5.0, 7.0], "num2": [-1.0, -0.5, 0.0, 0.0]}
    )
    background = _generate_background(df)
    expected_background = pd.DataFrame(
        {"num1": [3.0], "num2": [-0.25]}, columns=["num1", "num2"]
    )
    pd.testing.assert_frame_equal(expected_background, background)


def test_calculate_feature_importance(
    dummy_trainable_df, dummy_schema, dummy_model
):
    shap_values, shap_df = calculate_feature_importance(
        dummy_trainable_df, dummy_schema, dummy_model
    )
    # Check the shap values exist
    assert isinstance(shap_values, np.ndarray)
    # Check that shap_df has the right shape
    assert isinstance(shap_df, pd.Series)
    assert shap_df.shape == (7,)
