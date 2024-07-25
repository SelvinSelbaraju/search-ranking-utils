import pandas as pd
from search_ranking_utils.preprocessing.data_preprocessing import (
    create_imputations,
    impute_df,
    calculate_norm_stats,
    normalise_numerical_features,
    create_one_hot_encoder,
    one_hot_encode_categorical_features,
)


class Preprocessor:
    """
    Combine all preprocessing steps that are relevant to the schema
    A dataframe is also needed to learn the preprocessing values
    """

    def __init__(self, base_df: pd.DataFrame, schema: dict):
        self.base_df = base_df
        self.schema = schema

        # Optional Preprocessing objs
        # Imputation is not optional
        self.norm_stats = None
        self.one_hot_encoder = None
        self._create_objs()

    def _create_objs(self) -> dict:
        """
        Create all of the required objects for preprocessing
        """
        self.imputations = create_imputations(self.base_df, self.schema)
        if self.schema["features"].get("numerical"):
            self.norm_stats = calculate_norm_stats(self.base_df, self.schema)
        if self.schema["features"].get("categorical"):
            self.one_hot_encoder = create_one_hot_encoder(
                self.base_df, self.schema
            )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = impute_df(df, self.imputations)
        if self.norm_stats:
            df = normalise_numerical_features(df, self.norm_stats)
        if self.one_hot_encoder:
            df = one_hot_encode_categorical_features(
                df, self.schema, self.one_hot_encoder
            )
        return df
