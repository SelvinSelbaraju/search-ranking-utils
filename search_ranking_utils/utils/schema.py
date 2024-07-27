from typing import List, Optional, Any
from dataclasses import dataclass
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class ImputeStrategy:
    VALID_TYPES = ["mean", "median", "mode", "min", "max", "val"]

    def __init__(self, impute_type: str, val: Optional[Any] = None):
        self.impute_type = impute_type
        self.val = val

    def _check_type_valid(self, impute_type: str):
        if impute_type not in self.VALID_TYPES:
            raise ValueError(
                f"impute_type must be in {self.VALID_TYPES}, got {impute_type}"
            )


@dataclass
class Feature:
    """
    Store information on features
    """

    name: str
    impute_strategy: ImputeStrategy

    @classmethod
    def get_instance_from_config(cls, f_name, f_config: dict):
        return cls(
            name=f_name, impute_strategy=ImputeStrategy(**f_config["impute"])
        )


class CategoricalFeature(Feature):
    VALID_IMPUTE_TYPES = ["mode", "val"]

    def __init__(self, name: str, impute_strategy: ImputeStrategy):
        super().__init__(name, impute_strategy)
        if self.impute_strategy.impute_type not in self.VALID_IMPUTE_TYPES:
            raise ValueError(
                f"Categorical feature must have impute type "
                f"in {self.VALID_IMPUTE_TYPES}, "
                f"got {self.impute_strategy.impute_type}"
            )


class Schema:
    def __init__(
        self,
        target: str,
        query_col: str,
        categorical_features: List[Optional[CategoricalFeature]] = [],
        numerical_features: List[Optional[Feature]] = [],
    ):
        self.target = target
        self.query_col = query_col
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.all_features = categorical_features + numerical_features
        self.imputations = None
        self.normalisation_stats = None

    def set_imputations(self, df: pd.DataFrame) -> None:
        """
        Create a dictionary of feature to imputation value
        """
        imputations = {}
        for f in self.all_features:
            if f.impute_strategy.val:
                imputations[f.name] = f.impute_strategy.val
            else:
                impute_val = df[f.name].aggregate(
                    f.impute_strategy.impute_type
                )
                # Mode can be more than one value
                if f.impute_strategy.impute_type == "mode":
                    impute_val = impute_val[0]
                imputations[f.name] = impute_val
        self.imputations = imputations

    def set_norm_stats(self, df: pd.DataFrame) -> None:
        """
        Creates a dictionary for normalising features
        """
        norm_stats = {}
        for f in self.numerical_features:
            norm_stats[f.name] = {
                "mean": df[f.name].mean(),
                "std": df[f.name].std(),
            }
        self.norm_stats = norm_stats

    def set_vocabs(self, df: pd.DataFrame) -> None:
        """
        Creates a dictionary of categorical vocabs
        """
        vocabs = {}
        for f in self.categorical_features:
            # Don't want to include nan as a category
            vocabs[f.name] = list(
                df.sort_values(by=f.name)[f.name].dropna().unique()
            )
        self.vocabs = vocabs

    def set_stats(self, df: pd.DataFrame) -> None:
        self.set_imputations(df)
        self.set_norm_stats(df)
        self.set_vocabs(df)

    def get_model_features(self) -> List[str]:
        """
        In modelling, categorical cols get dropped
        Each categorical col is <col_name>_<cat_name>
        Requires vocab already set
        """
        if not hasattr(self, "vocabs"):
            raise NotImplementedError("Must set vocabs first")
        features = [f.name for f in self.numerical_features]
        for f in self.categorical_features:
            for category in self.vocabs[f]:
                features.append(f"{f.name}_{category}")
        return features

    @classmethod
    def create_instance_from_dict(cls, schema_dict: dict):
        logger.info(f"Creating schema using: {schema_dict}")
        categorical_features = []
        for f_name, f_config in (
            schema_dict["features"].get("categorical", {}).items()
        ):
            categorical_features.append(
                CategoricalFeature.get_instance_from_config(
                    f_name=f_name, f_config=f_config
                )
            )
        numerical_features = []
        for f_name, f_config in (
            schema_dict["features"].get("numerical", {}).items()
        ):
            numerical_features.append(
                Feature.get_instance_from_config(
                    f_name=f_name, f_config=f_config
                )
            )
        return cls(
            target=schema_dict["target"],
            query_col=schema_dict["query_col"],
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )
