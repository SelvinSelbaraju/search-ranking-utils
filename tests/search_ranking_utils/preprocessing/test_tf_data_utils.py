import pytest
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset
from search_ranking_utils.preprocessing.tf_data_utils import (
    create_tf_ds_from_dataframes,
)


@pytest.mark.parametrize(
    argnames=["test_batch_size", "expected_batch_size"],
    argvalues=[(None, 7), (2, 2)],
)
def test_create_tf_ds_from_dataframes(
    dummy_trainable_df, dummy_schema, test_batch_size, expected_batch_size
):
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    tf_ds = create_tf_ds_from_dataframes(X, y, test_batch_size)

    # Check that the data yields the right batch size
    for batch in tf_ds.take(1):
        assert batch[0]["u_c_f_1_loyal"].shape == (expected_batch_size)
        assert batch[1].shape[0] == expected_batch_size
