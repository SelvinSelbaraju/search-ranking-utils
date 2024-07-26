# Search Ranking Utils

This repository contains utils for building machine learning models to rank search results. The utils are packaged using Poetry.

## Setup

This setup is for testing the codebase on your machine. It assumes you are using a Mac.

### Dependencies

- Python 3.10 (can be configured using pyenv)
- Poetry 1.8.3
- libomp (via Brew)

### Instructions

1. Clone the repo. From the root of the directory, install all Python dependencies into a Poetry virtual env:

```
poetry install
```

## Repository Structure

The main code is in the `search_ranking_utils` directory.

- `eda`: Contains functions for exploring the data. Examples include checking key summary statistics (eg. missing values, data types, duplicate rows) and plotting distributions.
- `preprocessing`: Contains classes/functions to preprocess data (eg. imputing missing values, normalising data, encoding categorical feature) and prepare it for training and validation. Also includes some feature engineering code.
- `models`: Contains classes and functions to create models using Sklearn, XGBoost or Tensorflow.
- `evaluation`: Contains functions to evaluate models and plot useful information

All tests are in the `tests` directory.

## Dataset

To contextualise these utils and to be able to write unit tests for them, a dataset of dummy impressions with the following fields is created:

1. `query_id`: The identifier for the collection of results being ranked, not for the text of the query
1. `search_query`: The raw search query text
1. `user_id`: Dummy string identifier for a user
1. `local_timestamp`: The timestamp at which the impressions data was collected, in local time
1. `geo_location`: Dummy location where the user is
1. `u_n_f_1`: Dummy user numerical feature
1. `u_n_f_2`: Dummy user numerical feature
1. `u_c_f_1`: Dummy user categorical feature
1. `u_c_f_2`: Dummy user categorical feature
1. `product_id`: Dummy identifier for the product
1. `product_title`: Raw text title of the product
1. `product_description`: Raw text description of the product
1. `p_n_f_1`: Dummy product numerical feature
1. `p_n_f_2`: Dummy product numerical feature
1. `p_c_f_1`: Dummy product categorical feature
1. `p_c_f_2`: Dummy product categorical feature
1. `interacted`: A dummy target variable for whether the user had a positive interaction with this product
