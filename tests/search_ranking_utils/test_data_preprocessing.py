from search_ranking_utils.utils.testing import assert_dicts_equal
from search_ranking_utils.preprocessing.data_preprocessing import (
    create_imputations,
    impute_df,
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
    assert_dicts_equal(result, expected)


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
