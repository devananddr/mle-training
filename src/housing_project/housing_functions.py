import os
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_path):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """
    loading raw data from path.

    Called functions:
    -----------------
    None

    Parameters
    ----------
    housing_path : string
        path of raw data

    Returns
    -------
    housing_data : pandas.DataFrame
        pandas dataframe of raw data

    Raises
    ------
    None
    """

    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_dataset(housing):
    housing = housing
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    return train_set, train_set


def stratified_split_dataset(housing, test_size=0.2, random_state=42):
    """
    Performs stratified data splits.

    Called functions:
    -----------------
    None

    Parameters
    ----------
    housing : pandas.DataFrame
        raw data for housing prediction.
    test_size : float
        Define splitting proprtion b/w train and test data
    random_state : integer
        Define random state of splitting

    Returns
    -------
    housing_prepared : pandas.DataFrame
        dataframe with transformed data

    Raises
    ------
    None
    """

    housing = housing
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def get_corr_matrix(housing):
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    return corr_matrix


def data_transformation(housing):
    """
    Performs data transformations.

    Called functions:
    -----------------
    None

    Parameters
    ----------
    housing : pandas.DataFrame
        raw data for housing prediction.

    Returns
    -------
    housing_prepared : pandas.DataFrame
        dataframe with transformed data

    Raises
    ------
    None
    """

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    return housing_prepared


def score(housing_predictions, housing_labels):
    """
    Performs Evaluation of linear regression model.

    Called functions:
    -----------------
    None

    Parameters
    ----------
    housing_predictions : pandas.DataFrame
        DataFrame containing predicted data.
    housing_labels : pandas.DataFrame
        Dataframe contains Actual labels

    Returns
    -------
    score : python.dictionary
        dictionary contains all the evaluation scores

    Raises
    ------
    None
    """

    r_sq = r2_score(housing_predictions, housing_labels)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(housing_labels, housing_predictions)

    print("R2 Value: ", r_sq)
    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Error: ", mae)
    score = {"r2": r_sq, "mse": mse, "rmse": rmse, "mae": mae}

    return score


def linear_regression_model(train_df, train_label, test_df, test_label):
    """
    Performs linear regression model prediction.

    Called functions:
    -----------------
    None

    Parameters
    ----------
    train_df : pandas.DataFrame
        Input DataFrame containing training data.
    train_label : pandas.DataFrame
        Dataframe contains labels of the prediciton data
    test_df : pandas.DataFrame
        Input DataFrame containing testing data.
    test_label : pandas.DataFrame
        DataFrame containing labels of testing data..

    Returns
    -------
    Model_file
        Linear regression model file

    Raises
    ------
    None
    """
    lin_reg = LinearRegression()
    lin_reg.fit(train_df, train_label)

    return lin_reg
