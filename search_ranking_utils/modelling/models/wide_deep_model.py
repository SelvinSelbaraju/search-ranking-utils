from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import tensorflow as tf
from search_ranking_utils.modelling.tf_data_utils import (
    create_tfds_from_dataframes,
)


class WideDeepModel(tf.keras.Model):
    def __init__(
        self,
        hidden_units: List[int],
        wide_feature_names: Optional[List[str]] = [],
        deep_feature_names: Optional[List[str]] = [],
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
    ):
        super().__init__()
        self.wide_feature_names = wide_feature_names
        self.deep_feature_names = deep_feature_names
        self.hidden_units = hidden_units
        self.activation = activation
        self.kernel_initialiazer = kernel_initializer

        self._init_layers()

    def _get_inputs(
        self, x: Dict[str, tf.Tensor], features: List[str]
    ) -> tf.Tensor:
        """
        Take the input dict and turn it into a tensor
        """
        inputs = []
        for f in features:
            inputs.append(x[f])
        return tf.keras.layers.Concatenate()(inputs)

    def _init_layers(self) -> None:
        if self.wide_feature_names:
            self.wide_layers = [
                tf.keras.layers.Dense(
                    1,
                    kernel_initializer=self.kernel_initialiazer,
                )
            ]
        if self.deep_feature_names:
            self.deep_layers = [
                tf.keras.layers.Dense(
                    units,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initialiazer,
                )
                for units in self.hidden_units
            ]
            # For output
            self.deep_layers.append(
                tf.keras.layers.Dense(
                    1,
                    kernel_initializer=self.kernel_initialiazer,
                )
            )

    def call(self, x: Dict[str, tf.Tensor]) -> tf.Tensor:
        logits = []
        if self.wide_feature_names:
            wide_out = self._get_inputs(x, self.wide_feature_names)
            for layer in self.wide_layers:
                wide_out = layer(wide_out)
            logits.append(wide_out)
        if self.deep_feature_names:
            deep_out = self._get_inputs(x, self.deep_feature_names)
            for layer in self.deep_layers:
                deep_out = layer(deep_out)
            logits.append(deep_out)

        logits = tf.keras.layers.Add()(logits)
        return tf.keras.activations.sigmoid(logits)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        batch_size: int,
        epochs: int = 1,
        losses: List[tf.keras.losses.Loss] = [
            tf.keras.losses.BinaryCrossentropy()
        ],
        metrics: List[tf.keras.metrics.Metric] = [tf.keras.metrics.AUC()],
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.legacy.Adam(),  # noqa: E501
    ) -> None:
        """
        Sklearn like interface to train with
        """
        ds = create_tfds_from_dataframes(X, y, batch_size)
        self.compile(metrics=metrics, loss=losses, optimizer=optimizer)
        super().fit(ds, epochs=epochs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Run inference using a Dataframe
        Used to match sklearn interface
        """
        tf_X = {
            feature: tf.constant(data, shape=(len(X), 1))
            for feature, data in X.to_dict(orient="list").items()
        }
        scores = self.call(tf_X).numpy()
        return np.concatenate([1 - scores, scores], axis=1)
