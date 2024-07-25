import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_imputations(df: pd.DataFrame, schema: dict) -> dict:
    """
    Create a dictionary of feature to imputation value
    """
    imputations = {}
    for f, f_config in schema["features"]["categorical"].items():
        # Mode doesn't return a single value
        imputations[f] = df[f].aggregate(f_config["impute"]).values[0]
    for f, f_config in schema["features"]["numerical"].items():
        imputations[f] = df[f].aggregate(f_config["impute"])
    return imputations


def impute_df(df: pd.DataFrame, imputations: dict) -> pd.DataFrame:
    logger.info(f"Imputations: {imputations}")
    return df.fillna(imputations)


def calculate_norm_stats(df: pd.DataFrame, schema: dict) -> dict:
    """
    Calculate the mean and std for each feature
    This is so we can normalise the data
    """
    stats = {}
    for f in schema["features"]["numerical"]:
        stats[f] = {"mean": df[f].mean(), "std": df[f].std()}
    return stats


def normalise_numerical_features(
    df: pd.DataFrame, norm_stats: dict, epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Normalise features using pre-computed stats
    The `epsilon` param is to prevent zero division
    """
    for f, f_stats in norm_stats.items():
        f_mean = f_stats["mean"]
        f_std = epsilon if f_stats["std"] == 0 else f_stats["std"]
        df[f] = (df[f] - f_mean) / (f_std)
    return df
