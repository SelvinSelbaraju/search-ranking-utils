import pytest
import pandas as pd
from search_ranking_utils.utils.testing import assert_dicts_equal
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset
from search_ranking_utils.modelling.model_factory import ModelFactory
from search_ranking_utils.evaluation.metrics import (
    get_pointwise_metrics,
    calculate_query_ranking_metric,
    get_ranking_metrics,
)


@pytest.fixture
def predicted_df(dummy_trainable_df, dummy_schema) -> pd.DataFrame:
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    model = ModelFactory.get_instance_from_config(
        "sklearn.linear_model:LogisticRegression"
    )
    model.fit(X, y)
    df = df.assign(prediction=pd.Series(model.predict_proba(X)[:, 1]))
    return df


def test_get_pointwise_metrics(predicted_df, dummy_schema):
    metrics = get_pointwise_metrics(
        predicted_df, dummy_schema, "prediction", ["roc_auc_score", "log_loss"]
    )
    expected_metrics = {"roc_auc_score": 0.8, "log_loss": 0.5122374154189449}
    assert_dicts_equal(expected_metrics, metrics)


def test_calculate_query_ranking_metric():
    y_true = pd.Series([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    y_pred = pd.Series([[0.8, 0.6, 0.7], [0.9, 0.95, 0.3, 0.4]])
    ndcg = calculate_query_ranking_metric(y_true, y_pred, "ndcg_score")
    assert ndcg == 1.0


def test_get_ranking_metrics(predicted_df, dummy_schema):
    metrics = get_ranking_metrics(
        predicted_df, dummy_schema, "prediction", ["ndcg_score"]
    )
    expected_metrics = {"ndcg_score": 0.8154648767857287}
    assert_dicts_equal(expected_metrics, metrics)
