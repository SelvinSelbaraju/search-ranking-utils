import logging
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from search_ranking_utils.utils.schema import Schema

logger = logging.getLogger(__name__)


def impute_df(df: pd.DataFrame, imputations: dict) -> pd.DataFrame:
    logger.info(f"Imputations: {imputations}")
    return df.fillna(imputations)


def normalise_numerical_features(
    df: pd.DataFrame, norm_stats: dict, epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Normalise features using pre-computed stats
    The `epsilon` param is to prevent zero division
    """
    df = df.copy()
    logger.info(f"Norm stats: {norm_stats}")
    for f, f_stats in norm_stats.items():
        f_mean = f_stats["mean"]
        f_std = epsilon if f_stats["std"] == 0 else f_stats["std"]
        df[f] = (df[f] - f_mean) / (f_std)
    return df


def create_one_hot_encoder(
    df: pd.DataFrame, schema: Schema
) -> "OneHotEncoder":
    """
    Create a one hot encoder object to encode categorical features
    Needs to be used for preprocessing validation data as well
    """
    categorical_feature_names = [f.name for f in schema.categorical_features]
    # Don't want it to error on validation data with unknown category
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df[categorical_feature_names])
    return encoder


def one_hot_encode_categorical_features(
    df: pd.DataFrame, one_hot_encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Append one hot encoded features to the data
    Also drop original categorical feature
    """
    categorical_features = one_hot_encoder.feature_names_in_
    categorical_data = df[categorical_features]
    # Create the OHE columns
    columns = []
    for i in range(len(categorical_features)):
        f = categorical_features[i]
        for category in one_hot_encoder.categories_[i]:
            columns.append(f"{f}_{category}")
    encoded = pd.DataFrame(
        one_hot_encoder.transform(categorical_data),
        columns=columns,
        index=df.index,
    )
    # Drop original categorical features and concat on
    df = df.drop(categorical_features, axis=1)
    return pd.concat([df, encoded], axis=1)


def drop_cols(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """
    Drop columns not specified in the schema
    """
    features = [f.name for f in schema.all_features]
    features.append(schema.target)
    features.append(schema.query_col)
    return df[features]


def split_dataset(
    df: pd.DataFrame, schema: Schema
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return the original df, X and y
    Assumes data already preprocessed
    """
    X = df[schema.get_model_features()]
    y = df[schema.target]
    return df, X, y
