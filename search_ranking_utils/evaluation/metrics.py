from typing import List
import pandas as pd
from importlib import import_module

SKLEARN_METRICS_MODULE = import_module("sklearn.metrics")


def get_pointwise_metrics(
    df: pd.DataFrame,
    schema: dict,
    y_pred_col: str,
    pointwise_metrics: List[str],
):
    """
    Calculate all of the desired pointwise metrics on y_true and y_pred
    """
    metrics = {}
    for metric in pointwise_metrics:
        metric_func = getattr(SKLEARN_METRICS_MODULE, metric)
        metrics[metric] = metric_func(df[schema["target"]], df[y_pred_col])
    return metrics
