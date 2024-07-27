from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
import pytest
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset
from search_ranking_utils.modelling.tf_data_utils import (
    create_tfds_from_dataframes,
)
from search_ranking_utils.modelling.models.wide_deep_model import WideDeepModel


@pytest.fixture
def test_model(dummy_schema):
    deep_feature_names = [f.name for f in dummy_schema.numerical_features]
    wide_feature_names = [
        f
        for f in dummy_schema.get_model_features()
        if f not in deep_feature_names
    ]
    model = WideDeepModel(
        hidden_units=[4, 2, 2],
        wide_feature_names=wide_feature_names,
        deep_feature_names=deep_feature_names,
    )
    return model


@pytest.fixture
def dummy_tfds(dummy_trainable_df, dummy_schema) -> tf.data.Dataset:
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    ds = create_tfds_from_dataframes(X=X, y=y)
    return ds


@pytest.fixture
def dummy_tfds_tuple(dummy_tfds) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    for ds_tuple in dummy_tfds.take(1):
        return ds_tuple


def test_wide_deep_model_init(test_model):
    assert test_model.deep_feature_names == ["u_n_f_2", "p_n_f_1"]
    assert test_model.wide_feature_names == [
        "u_c_f_1_infrequent",
        "u_c_f_1_loyal",
        "p_c_f_2_cooking",
        "p_c_f_2_food",
        "p_c_f_2_jacket",
        "p_c_f_2_kids",
    ]
    assert test_model.hidden_units == [4, 2, 2]
    assert test_model.activation == "relu"
    assert test_model.kernel_initialiazer == "he_normal"


def test_wide_model_init_layers(test_model):
    assert test_model.wide_layers[0].units == 1
    assert test_model.deep_layers[0].units == 4
    assert test_model.deep_layers[1].units == 2
    assert test_model.deep_layers[2].units == 2


def test_wide_deep_model_get_inputs(dummy_tfds_tuple, test_model):
    deep_inputs = test_model._get_inputs(
        dummy_tfds_tuple[0], ["u_n_f_2", "p_n_f_1"]
    )
    assert isinstance(deep_inputs, tf.Tensor)
    assert deep_inputs.shape == (7, 2)


def test_wide_deep_model_call(dummy_tfds_tuple, test_model):
    output = test_model(dummy_tfds_tuple[0])
    assert output.shape == (7, 1)


def test_wide_deep_model_fit(dummy_trainable_df, dummy_schema, test_model):
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    # See if output changes after training
    untrained_output = test_model.predict_proba(X)[:, 1]
    test_model.fit(X, y, 2)
    trained_output = test_model.predict_proba(X)[:, 1]
    # Make sure sum of equal less than overall length
    assert np.equal(untrained_output, trained_output).sum() < 7


def test_wide_deep_model_predict_proba(
    dummy_trainable_df, dummy_schema, test_model
):
    df, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    output = test_model.predict_proba(X)
    assert output[:, 1].shape == (7,)
