from typing import List, Dict
import numpy as np
import pandas as pd
from importlib import import_module
from search_ranking_utils.utils.schema import Schema

SKLEARN_METRICS_MODULE = import_module("sklearn.metrics")


def get_pointwise_metrics(
    df: pd.DataFrame,
    schema: Schema,
    y_pred_col: str,
    pointwise_metrics: List[str],
) -> Dict[str, float]:
    """
    Calculate all of the desired pointwise metrics on y_true and y_pred
    """
    metrics = {}
    for metric in pointwise_metrics:
        metric_func = getattr(SKLEARN_METRICS_MODULE, metric)
        metrics[metric] = metric_func(df[schema.target], df[y_pred_col])
    return metrics


def calculate_query_ranking_metric(
    y_true: pd.Series, y_pred: pd.Series, ranking_metric_name: str
) -> float:
    """
    For a single query, calculate a ranking metric
    Needs to be done query by query as variable length queries
    Each pd.Series contains an array of actual/predicted scores
    Then average the query scores at the end
    """
    metric_func = getattr(SKLEARN_METRICS_MODULE, ranking_metric_name)
    query_metrics = np.array([])
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    for i in range(len(y_true)):
        # Needs to be 2-D for ranking metric calculation
        y_true_arr = np.array(y_true[i]).reshape(1, -1)
        y_pred_arr = np.array(y_pred[i]).reshape(1, -1)
        query_metrics = np.append(
            query_metrics, metric_func(y_true_arr, y_pred_arr)
        )
    return query_metrics.mean()


def get_ranking_metrics(
    df: pd.DataFrame,
    schema: Schema,
    y_pred_col: str,
    ranking_metrics: List[str],
) -> Dict[str, float]:
    metrics = {}
    y_true_queries = df.groupby(schema.query_col)[schema.target].agg(list)
    # Remove all queries of size 1 as can't rank
    # Also remove any queries which had no relevant items
    drop_indices = (y_true_queries.apply(lambda x: len(x) > 1)) & (
        y_true_queries.apply(lambda x: sum(x) > 0)
    )
    y_true_queries = y_true_queries[drop_indices]
    y_pred_queries = df.groupby(schema.query_col)[y_pred_col].agg(list)[
        drop_indices
    ]
    for metric in ranking_metrics:
        metrics[metric] = calculate_query_ranking_metric(
            y_true_queries, y_pred_queries, metric
        )
    return metrics
