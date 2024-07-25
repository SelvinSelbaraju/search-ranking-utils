import os
import pytest
import pandas as pd
from search_ranking_utils.utils.files import load_json

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
def dummy_schema() -> dict:
    return load_json(DUMMY_SCHEMA_PATH)
