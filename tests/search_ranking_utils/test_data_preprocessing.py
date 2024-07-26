import pandas as pd
from search_ranking_utils.utils.testing import assert_dicts_equal
from search_ranking_utils.preprocessing.data_preprocessing import (
    create_imputations,
    impute_df,
    calculate_norm_stats,
    normalise_numerical_features,
    create_one_hot_encoder,
    one_hot_encode_categorical_features,
    drop_cols,
    split_dataset,
)


# Create imputations
def test_create_imputations(dummy_df, dummy_schema):
    result = create_imputations(dummy_df, dummy_schema)
    expected = {
        "u_c_f_1": "infrequent",
        "p_c_f_2": "food",
        "u_n_f_2": -500.5,
        "p_n_f_1": 100.50,
    }
    assert_dicts_equal(expected, result)


# Impute data using imputations
def test_impute_df(dummy_df, dummy_schema):
    imputations = create_imputations(dummy_df, dummy_schema)
    imputed_df = impute_df(dummy_df, imputations)
    # Check specific rows of data and features
    # query_id,product_id is unique in dummy data
    # Intentionally left nulls there
    assert (
        imputed_df[
            (imputed_df["query_id"] == "query2")
            & (imputed_df["product_id"] == "prod102")
        ]["u_c_f_1"].values[0]
        == "infrequent"
    )
    assert (
        imputed_df[
            (imputed_df["query_id"] == "query2")
            & (imputed_df["product_id"] == "prod103")
        ]["p_c_f_2"].values[0]
        == "food"
    )


def test_calculate_norm_stats(dummy_df, dummy_schema):
    result = calculate_norm_stats(dummy_df, dummy_schema)
    expected = {
        "u_n_f_2": {"mean": -500.5, "std": 547.1748349476609},
        "p_n_f_1": {"mean": 107.60928571428572, "std": 92.39643014517083},
    }
    assert_dicts_equal(expected, result)


def test_normalise_numerical_features(dummy_df, dummy_schema):
    stats = calculate_norm_stats(dummy_df, dummy_schema)
    result = normalise_numerical_features(dummy_df, stats)
    # Check 1 of the numerical features in different rows
    assert result.iloc[0]["u_n_f_2"] == 0.912870929175277
    assert result.iloc[5]["p_n_f_1"] == -1.148413261719848


def test_one_hot_encode_categorical_features(dummy_df, dummy_schema):
    encoder = create_one_hot_encoder(dummy_df, dummy_schema)
    result = one_hot_encode_categorical_features(
        dummy_df, dummy_schema, encoder
    )
    # Check if each of the OHE columns exists
    # Check if got the right values for two rows of data
    assert result.iloc[0]["u_c_f_1_loyal"] == 1
    assert result.iloc[3]["u_c_f_1_loyal"] == 0
    assert result.iloc[0]["u_c_f_1_infrequent"] == 0
    assert result.iloc[3]["u_c_f_1_infrequent"] == 1

    assert result.iloc[0]["p_c_f_2_jacket"] == 1
    assert result.iloc[3]["p_c_f_2_jacket"] == 0
    assert result.iloc[0]["p_c_f_2_kids"] == 0
    assert result.iloc[3]["p_c_f_2_kids"] == 0
    assert result.iloc[0]["p_c_f_2_food"] == 0
    assert result.iloc[3]["p_c_f_2_food"] == 1
    assert result.iloc[0]["p_c_f_2_cooking"] == 0
    assert result.iloc[3]["p_c_f_2_cooking"] == 0

    # Check if old categorical columns are dropped
    assert "u_c_f_1" not in result.columns
    assert "p_c_f_2" not in result.columns


def test_drop_cols(dummy_df, dummy_schema):
    expected = dummy_df[
        ["u_c_f_1", "p_c_f_2", "u_n_f_2", "p_n_f_1", "interacted", "query_id"]
    ]
    result = drop_cols(dummy_df, dummy_schema)
    pd.testing.assert_frame_equal(expected, result)


def test_split_dataset(dummy_df, dummy_schema):
    drop_df = drop_cols(dummy_df, dummy_schema)
    df, X, y = split_dataset(drop_df, dummy_schema)

    # Check df is not changed
    pd.testing.assert_frame_equal(drop_df, df)

    # Check X has the right columns
    assert sorted(X.columns) == [
        "p_c_f_2",
        "p_n_f_1",
        "u_c_f_1",
        "u_n_f_2",
    ]

    # Check y has the right values
    expected_y = pd.Series([1, 0, 0, 0, 1, 0, 0], name="interacted")
    pd.testing.assert_series_equal(expected_y, y)
