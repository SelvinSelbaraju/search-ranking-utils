import pandas as pd
from search_ranking_utils.preprocessing.data_preprocessing import (
    impute_df,
    normalise_numerical_features,
    map_oov_categories,
    create_one_hot_encoder,
    one_hot_encode_categorical_features,
    drop_cols,
    split_dataset,
)


# Impute data using imputations
def test_impute_df(dummy_df):
    imputations = {
        "u_c_f_1": "infrequent",
        "p_c_f_2": "food",
        "u_n_f_2": -500.5,
        "p_n_f_1": -50.0,
    }
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


def test_normalise_numerical_features(dummy_df):
    stats = {
        "u_n_f_2": {"mean": -500.5, "std": 547.1748349476609},
        "p_n_f_1": {"mean": 107.60928571428572, "std": 92.39643014517083},
    }
    result = normalise_numerical_features(dummy_df, stats)
    # Check 1 of the numerical features in different rows
    assert result.iloc[0]["u_n_f_2"] == 0.912870929175277
    assert result.iloc[5]["p_n_f_1"] == -1.148413261719848


def test_map_oov_categories(dummy_df, dummy_schema):
    # Impute first don't need to deal with nulls
    df = impute_df(dummy_df, dummy_schema.imputations)
    df = map_oov_categories(df, dummy_schema)
    # Check that the oov categories are no longer present
    assert sorted(df["p_c_f_2"].unique()) == ["<OTHER>", "food", "jacket"]
    # Check other columns are unaltered
    assert sorted(df["u_c_f_1"].unique()) == ["infrequent", "loyal"]


def test_one_hot_encode_categorical_features(dummy_df, dummy_schema):
    df = map_oov_categories(dummy_df, dummy_schema)
    encoder = create_one_hot_encoder(df, dummy_schema)
    result = one_hot_encode_categorical_features(df, encoder)
    # Check if each of the OHE columns exists
    # Check if got the right values for two rows of data
    assert result.iloc[0]["u_c_f_1_loyal"] == 1
    assert result.iloc[3]["u_c_f_1_loyal"] == 0
    assert result.iloc[0]["u_c_f_1_infrequent"] == 0
    assert result.iloc[3]["u_c_f_1_infrequent"] == 1

    assert result.iloc[0]["p_c_f_2_jacket"] == 1
    assert result.iloc[3]["p_c_f_2_jacket"] == 0
    assert result.iloc[0]["p_c_f_2_<OTHER>"] == 0
    assert result.iloc[3]["p_c_f_2_<OTHER>"] == 0
    assert result.iloc[0]["p_c_f_2_food"] == 0
    assert result.iloc[3]["p_c_f_2_food"] == 1

    # Check if old categorical columns are dropped
    assert "u_c_f_1" not in result.columns
    assert "p_c_f_2" not in result.columns


def test_drop_cols(dummy_df, dummy_schema):
    expected = dummy_df[
        ["u_c_f_1", "p_c_f_2", "u_n_f_2", "p_n_f_1", "interacted", "query_id"]
    ]
    result = drop_cols(dummy_df, dummy_schema)
    pd.testing.assert_frame_equal(expected, result)


def test_split_dataset(dummy_trainable_df, dummy_schema):
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)

    # Check df is not changed
    pd.testing.assert_frame_equal(dummy_trainable_df, df)

    # Check X has the right columns
    assert sorted(X.columns) == [
        "p_c_f_2_<OTHER>",
        "p_c_f_2_food",
        "p_c_f_2_jacket",
        "p_n_f_1",
        "u_c_f_1_infrequent",
        "u_c_f_1_loyal",
        "u_n_f_2",
    ]

    # Check y has the right values
    expected_y = pd.Series([1, 0, 0, 0, 1, 0, 0], name="interacted")
    pd.testing.assert_series_equal(expected_y, y)
