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
    """
    tf_X = {
        feature: tf.constant(data)
        for feature, data in X.to_dict(orient="list").items()
    }
    tf_y = tf.constant(y.values)
    ds = tf.data.Dataset.from_tensor_slices((tf_X, tf_y))
    if batch_size:
        ds = ds.batch(batch_size)
    else:
        ds = ds.batch(len(X))
    # Need to have 2 dimensions
    ds = ds.map(
        lambda x, y: (
            {k: tf.expand_dims(v, axis=1) for k, v in x.items()},
            tf.expand_dims(y, axis=1),
        )
    )
    return ds
