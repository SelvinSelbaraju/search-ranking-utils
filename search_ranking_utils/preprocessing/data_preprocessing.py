import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_imputations(df: pd.DataFrame, schema: dict) -> dict:
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
