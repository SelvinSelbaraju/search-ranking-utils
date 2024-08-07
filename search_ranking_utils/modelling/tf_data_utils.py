from typing import Optional
import pandas as pd
import tensorflow as tf


def create_tfds_from_dataframes(
    X: pd.DataFrame, y: pd.Series, batch_size: Optional[int] = None
) -> tf.data.Dataset:
    """
    Create a TF Dataset from a Dataframe
    If no batch size provided, combine into single batch
    Useful for running inference on validation data
    The tensors need to have two dimensions
    """
    tf_X = {
        feature: tf.constant(data, shape=(len(X), 1))
        for feature, data in X.to_dict(orient="list").items()
    }
    tf_y = tf.constant(y.values, shape=(len(y), 1))
    ds = tf.data.Dataset.from_tensor_slices((tf_X, tf_y))
    if batch_size:
        ds = ds.batch(batch_size)
    else:
        ds = ds.batch(len(X))
    return ds
