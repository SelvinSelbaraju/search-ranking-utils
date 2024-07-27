import tensorflow as tf
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset
from search_ranking_utils.modelling.tf_data_utils import (
    create_tfds_from_dataframes,
)


def test_create_tfds_from_dataframes(dummy_trainable_df, dummy_schema):
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    ds = create_tfds_from_dataframes(X, y, 2)
    # Check a batch
    for batch in ds.take(1):
        test_batch = batch
    # Check types
    assert isinstance(test_batch[0], dict)
    assert isinstance(test_batch[1], tf.Tensor)
    # Check data
    assert isinstance(test_batch[0]["u_n_f_2"], tf.Tensor)
    assert test_batch[0]["u_n_f_2"].shape == (2)
    assert test_batch[1].shape == (2)
