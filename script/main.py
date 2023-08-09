import json
import logging
import logging.config
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from housing_project import housing_functions

# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5006
remote_server_uri = "http://127.0.0.1:5006"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

mlflow.tracking.get_tracking_uri()

exp_name = "model_dev_cycle"
mlflow.set_experiment(exp_name)

with open("../config.json") as config_file:
    config = json.load(config_file)

HOUSING_PATH = config["raw_data_path"]

logger = logging.getLogger()

with mlflow.start_run():
    # To download the file and save it to the path

    housing_functions.fetch_housing_data(HOUSING_PATH)

    # loading the dataset
    housing = housing_functions.load_housing_data(HOUSING_PATH)

    # Splitting the dataset to train and test
    test_size = config["test_size"]
    housing_train, housing_test = housing_functions.stratified_split_dataset(
        housing, test_size=test_size, random_state=42
    )

    # Applying the transformation to train and test datasets
    housing_train_transformed = housing_functions.data_transformation(housing_train)
    housing_test_transformed = housing_functions.data_transformation(housing_test)
    logger.info("Data transformation completed")

    # Saving the Processed datasets
    process_data_path = os.path.join("..", "data", "processed")
    housing_train_transformed.to_csv(
        os.path.join(process_data_path, "train.csv"), index=False
    )
    housing_test_transformed.to_csv(
        os.path.join(process_data_path, "test.csv"), index=False
    )
    logger.info(f"Processed data saved in {process_data_path}")

    mlflow.log_param(key="raw_data_path", value=HOUSING_PATH)
    mlflow.log_param(key="proccesed_data_path", value=process_data_path)
    mlflow.log_artifact(HOUSING_PATH)
    mlflow.log_artifact(process_data_path)
    # print("Save to: {}".format(mlflow.get_artifact_uri()))

    # reading train and test data
    train_path = config["process_data_path"] + "train.csv"
    test_path = config["process_data_path"] + "test.csv"

    # model output path
    output_path = config["model_path"]

    # Loading the test and train datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.info("Train and Test data read successfully")

    # splitting x and y of the datasets
    x_train = train_df.drop("median_house_value", axis=1)
    y_train = train_df[["median_house_value"]]
    x_test = test_df.drop("median_house_value", axis=1)
    y_test = test_df[["median_house_value"]]

    # Applying linear regression model
    model_1 = housing_functions.linear_regression_model(
        x_train, y_train, x_test, y_test
    )

    with open(output_path + "model.pkl", "wb") as f:
        pickle.dump(model_1, f)
    logger.info("Model pickle stored in artifacts")

    mlflow.log_param(key="model path", value=output_path)
    mlflow.log_artifact(output_path)
    # print("Save to: {}".format(mlflow.get_artifact_uri()))

    model_path = config["model_path"]
    test_data_path = config["process_data_path"] + "test.csv"

    # reading test dataset
    test_df = pd.read_csv(test_data_path)
    logger.info("Test data read successfully ")

    with open(model_path + "model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully ")

    # splitting x and y of the dataset
    x_test = test_df.drop("median_house_value", axis=1)
    y_test = test_df[["median_house_value"]]

    predictions = model.predict(x_test)

    # Getting the scores of the model
    score = housing_functions.score(predictions, y_test)
    logger.info(f"R_Squared value : {score['r2']}")
    logger.info(f"Mean Squared Error : {score['mse']}")
    logger.info(f"Root Mean Squared Error : {score['rmse']}")
    logger.info(f"Mean Absolute Error : {score['mae']}")

    mlflow.log_metric(key="rmse", value=score["rmse"])
    mlflow.log_metric(key="mse", value=score["mse"])
    mlflow.log_metrics({"mae": score["mae"], "r2": score["r2"]})
    print("Save to: {}".format(mlflow.get_artifact_uri()))
