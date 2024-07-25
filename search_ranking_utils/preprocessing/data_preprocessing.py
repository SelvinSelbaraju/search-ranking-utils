import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
    logger.info(f"Norm stats: {norm_stats}")
    for f, f_stats in norm_stats.items():
        f_mean = f_stats["mean"]
        f_std = epsilon if f_stats["std"] == 0 else f_stats["std"]
        df[f] = (df[f] - f_mean) / (f_std)
    return df


def create_one_hot_encoder(df: pd.DataFrame, schema: dict) -> "OneHotEncoder":
    """
    Create a one hot encoder object to encode categorical features
    Needs to be used for preprocessing validation data as well
    """
    categorical_features = list(schema["features"]["categorical"].keys())
    categorical_data = df[categorical_features]
    # Don't want it to error on validation data with unknown category
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(categorical_data)
    return encoder


def one_hot_encode_categorical_features(
    df: pd.DataFrame, schema: dict
) -> pd.DataFrame:
    """
    Append one hot encoded features to the data
    Also drop original categorical feature
    """
    categorical_features = list(schema["features"]["categorical"].keys())
    categorical_data = df[categorical_features]
    one_hot_encoder = create_one_hot_encoder(df, schema)
    # Create the OHE columns
    columns = []
    for i in range(len(categorical_features)):
        f = categorical_features[i]
        for category in one_hot_encoder.categories_[i]:
            columns.append(f"{f}_{category}")
    encoded = pd.DataFrame(
        one_hot_encoder.transform(categorical_data), columns=columns
    )
    # Drop original categorical features and concat on
    df = df.drop(categorical_features, axis=1)
    return pd.concat([df, encoded], axis=1)
