import os
import pytest
import pandas as pd
from search_ranking_utils.utils.files import load_json
from search_ranking_utils.utils.schema import Schema
from search_ranking_utils.preprocessing.preprocessor import Preprocessor

"""
Contains reusable artifacts for tests like data and config
"""

LOCAL_DIRECTORY = os.path.dirname(__file__)
DUMMY_CSV_PATH = os.path.join(LOCAL_DIRECTORY, "data", "dummy_data.csv")
DUMMY_SCHEMA_PATH = os.path.join(LOCAL_DIRECTORY, "config", "schema.json")


@pytest.fixture(scope="session")
def dummy_df() -> pd.DataFrame:
    df = pd.read_csv(DUMMY_CSV_PATH)
    return df


@pytest.fixture(scope="session")
def dummy_schema(dummy_df) -> Schema:
    schema_dict = load_json(DUMMY_SCHEMA_PATH)
    schema = Schema.create_instance_from_dict(schema_dict)
    schema.set_stats(dummy_df)
    return schema


@pytest.fixture(scope="session")
def dummy_trainable_df(dummy_df, dummy_schema) -> pd.DataFrame:
    preprocessor = Preprocessor(dummy_df, dummy_schema)
    return preprocessor(dummy_df)
