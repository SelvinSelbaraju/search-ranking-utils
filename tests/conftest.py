import os
import pytest
import pandas as pd

"""
Contains reusable artifacts for tests like data and config
"""

LOCAL_DIRECTORY = os.path.dirname(__file__)
DUMMY_CSV_PATH = os.path.join(LOCAL_DIRECTORY, "data", "dummy_data.csv")


@pytest.fixture(scope="session")
def dummy_df() -> pd.DataFrame:
    df = pd.read_csv(DUMMY_CSV_PATH)
    return df
