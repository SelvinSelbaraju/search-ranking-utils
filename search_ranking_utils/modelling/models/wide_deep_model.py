from typing import Dict, List, Optional
import tensorflow as tf


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
                    activation=self.activation,
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
                    activation=self.activation,
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
