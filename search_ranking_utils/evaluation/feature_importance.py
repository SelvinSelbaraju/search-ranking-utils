from typing import Tuple
import numpy as np
import pandas as pd
import shap
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset


def calculate_feature_importance(
    df: pd.DataFrame, schema: dict, model
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Given a preprocessed dataset, calcualte feature importance
    """
    df, X, y = split_dataset(df, schema)

    # Use Shap kernel predict by defining inner function
    def f(X):
        return model.predict_proba(X)[:, 1]

    # Used as the background values to sample from
    background = _generate_background(X)

    explainer = shap.Explainer(f, background)
    shap_values = explainer(X).values
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df = shap_df.mean().sort_values(ascending=False)
    return shap_values, shap_df


def _generate_background(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate background samples using medians
    Assumes df is preprocessed, no categorical features
    """
    background = df.head(1)
    background[df.columns] = df.median()
    return background
