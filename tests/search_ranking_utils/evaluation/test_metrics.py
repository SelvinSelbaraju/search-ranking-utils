import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from search_ranking_utils.utils.testing import assert_dicts_equal
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset
from search_ranking_utils.modelling.model_factory import ModelFactory
from search_ranking_utils.evaluation.metrics import get_pointwise_metrics


@pytest.fixture
def trained_model(dummy_trainable_df, dummy_schema) -> LogisticRegression:
    _, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    model = ModelFactory.get_instance_from_cofing(
        "sklearn.linear_model:LogisticRegression"
    )
    model.fit(X, y)
    return model


def test_get_pointwise_metrics(
    dummy_trainable_df, dummy_schema, trained_model
):
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    df["prediction"] = pd.Series(trained_model.predict_proba(X)[:, 1])
    metrics = get_pointwise_metrics(
        df, dummy_schema, "prediction", ["roc_auc_score", "log_loss"]
    )
    expected_metrics = {"roc_auc_score": 0.8, "log_loss": 0.5122374154189449}
    assert_dicts_equal(expected_metrics, metrics)
