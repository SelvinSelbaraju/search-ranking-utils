from sklearn.preprocessing import OneHotEncoder
from search_ranking_utils.utils.testing import assert_dicts_equal
from search_ranking_utils.preprocessing.preprocessor import Preprocessor


def test_preprocessor_init(dummy_df, dummy_schema):
    preprocessor = Preprocessor(dummy_df, dummy_schema)
    assert_dicts_equal(
        {
            "u_c_f_1": "infrequent",
            "p_c_f_2": "food",
            "u_n_f_2": -500.5,
            "p_n_f_1": -50.0,
        },
        preprocessor.schema.imputations,
    )
    assert_dicts_equal(
        {
            "u_n_f_2": {"mean": -500.5, "std": 547.1748349476609},
            "p_n_f_1": {"mean": 107.60928571428572, "std": 92.39643014517083},
        },
        preprocessor.schema.norm_stats,
    )

    assert isinstance(preprocessor.one_hot_encoder, OneHotEncoder)


def test_preprocessor_call(dummy_df, dummy_schema):
    preprocessor = Preprocessor(dummy_df, dummy_schema)
    result = preprocessor(dummy_df)
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
