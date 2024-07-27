import pandas as pd
from search_ranking_utils.utils.schema import Schema
from search_ranking_utils.preprocessing.data_preprocessing import (
    impute_df,
    normalise_numerical_features,
    create_one_hot_encoder,
    drop_cols,
    one_hot_encode_categorical_features,
)


class Preprocessor:
    """
    Combine all preprocessing steps that are relevant to the schema
    A dataframe is also needed to learn the preprocessing values
    """

    def __init__(self, base_df: pd.DataFrame, schema: Schema):
        self.base_df = base_df
        self.schema = schema

        # Optional Preprocessing objs
        if len(self.schema.categorical_features) > 0:
            self.one_hot_encoder = create_one_hot_encoder(
                self.base_df, self.schema
            )
        else:
            self.one_hot_encoder = None

    def __call__(
        self, df: pd.DataFrame, drop_redundant: bool = True
    ) -> pd.DataFrame:
        if drop_redundant:
            df = drop_cols(df, self.schema)
        df = impute_df(df, self.schema.imputations)
        if len(self.schema.numerical_features) > 0:
            df = normalise_numerical_features(df, self.schema.norm_stats)
        if self.one_hot_encoder:
            df = one_hot_encode_categorical_features(df, self.one_hot_encoder)
        return df
