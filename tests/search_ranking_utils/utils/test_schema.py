import pytest
from search_ranking_utils.utils.testing import assert_dicts_equal
from search_ranking_utils.utils.schema import Schema


@pytest.fixture
def dummy_schema_obj(dummy_schema):
    return Schema.create_instance_from_dict(dummy_schema)


def test_schema_create_instance_from_dict(dummy_schema_obj):
    # Check categorical feature names
    expected_categorical = ["u_c_f_1", "p_c_f_2"]
    result_categorical = [
        f.name for f in dummy_schema_obj.categorical_features
    ]
    assert expected_categorical == result_categorical
    # Check numerical features
    expected_numerical = ["u_n_f_2", "p_n_f_1"]
    result_numerical = [f.name for f in dummy_schema_obj.numerical_features]
    assert expected_numerical == result_numerical
    # Check other fields
    assert dummy_schema_obj.target == "interacted"
    assert dummy_schema_obj.query_col == "query_id"


def test_set_imputations(dummy_df, dummy_schema_obj):
    dummy_schema_obj.set_imputations(dummy_df)
    expected = {
        "u_c_f_1": "infrequent",
        "p_c_f_2": "food",
        "u_n_f_2": -500.5,
        "p_n_f_1": 100.50,
    }
    assert_dicts_equal(expected, dummy_schema_obj.imputations)


def test_set_norm_stats(dummy_df, dummy_schema_obj):
    dummy_schema_obj.set_norm_stats(dummy_df)
    expected = {
        "u_n_f_2": {"mean": -500.5, "std": 547.1748349476609},
        "p_n_f_1": {"mean": 107.60928571428572, "std": 92.39643014517083},
    }
    assert_dicts_equal(expected, dummy_schema_obj.norm_stats)


def test_set_vocabs(dummy_df, dummy_schema_obj):
    dummy_schema_obj.set_vocabs(dummy_df)
    expected = {
        "u_c_f_1": ["infrequent", "loyal"],
        "p_c_f_2": ["cooking", "food", "jacket", "kids"],
    }
    assert_dicts_equal(expected, dummy_schema_obj.vocabs)
