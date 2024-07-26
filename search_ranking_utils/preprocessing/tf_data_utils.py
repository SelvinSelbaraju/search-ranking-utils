from typing import Optional
import pandas as pd
import tensorflow as tf


def create_tf_ds_from_dataframes(
    X: pd.DataFrame, y: pd.Series, batch_size: Optional[int] = None
) -> tf.data.Dataset:
    """
    Create a Tensorflow dataset from X and y
    A Tensorflow dataset is an iterator of tuples
    First elem is dict of feature to feature values
    Second elem is target

    Batches data if batch provided, otherwise batches into a single batch
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
    return ds
