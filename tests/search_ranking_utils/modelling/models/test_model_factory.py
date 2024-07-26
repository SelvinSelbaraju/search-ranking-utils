import pytest
import sklearn.ensemble
import sklearn.linear_model
from search_ranking_utils.modelling.models.model_factory import ModelFactory
from search_ranking_utils.preprocessing.data_preprocessing import split_dataset


@pytest.mark.parametrize(
    argnames=["test_package_module_cls", "test_model_config", "expected_cls"],
    argvalues=[
        (
            "sklearn.linear_model:LogisticRegression",
            {"C": 0.5},
            sklearn.linear_model.LogisticRegression,
        ),
        (
            "sklearn.ensemble:RandomForestClassifier",
            {"n_estimators": 50, "min_samples_split": 10},
            sklearn.ensemble.RandomForestClassifier,
        ),
    ],
)
def test_model_factory_get_instance_from_config(
    test_package_module_cls, test_model_config, expected_cls
):
    model_instance = ModelFactory.get_instance_from_cofing(
        test_package_module_cls,
        test_model_config,
    )
    # Check right instance
    assert isinstance(
        model_instance, expected_cls
    ), f"Expected {expected_cls}, got {type(model_instance)}"

    # Check model config
    for param, param_val in test_model_config.items():
        model_instance_param = getattr(model_instance, param)
        assert (
            model_instance_param == param_val
        ), f"Expected {param_val}, got {model_instance_param}"


def test_model_factory_fit(dummy_trainable_df, dummy_schema):
    model = ModelFactory.get_instance_from_cofing(
        "sklearn.linear_model:LogisticRegression",
        {"C": 0.5},
    )
    _, X, y = split_dataset(dummy_trainable_df, dummy_schema)
    model.fit(X, y)
