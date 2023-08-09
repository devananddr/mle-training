import argparse
import json
import logging
import logging.config
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd

from housing_project import housing_functions

# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.1 --port 5006
remote_server_uri = "http://127.0.0.1:5006"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

exp_name = "model_training"
mlflow.set_experiment(exp_name)


with open("../config.json") as config_file:
    config = json.load(config_file)


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="INFO"
):
    logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    help="Specify the path for train and test datasets (train.csv, test.csv - both with labels)",
)
parser.add_argument(
    "-o", "--output", help="Specify the path for dumping the model/pickle file "
)
parser.add_argument(
    "-lp",
    "--log-path",
    help="Specify if loggin required in the log file ",
    action="store_true",
)
parser.add_argument(
    "-ll",
    "--log-level",
    help="Specify the log level [INFO, DEBUG, WARNING, ERROR, CRITICAL] ",
)
parser.add_argument(
    "-ncl", "--no-console-log", help="To remove console logging", action="store_false"
)
args = parser.parse_args()
arg_input = args.input
arg_output = args.output
arg_log_path = args.log_path
arg_log_level = args.log_level
arg_no_console_log = args.no_console_log


if arg_input:
    train_path = arg_input + "train.csv"
    test_path = arg_input + "test.csv"
else:
    train_path = config["process_data_path"] + "train.csv"
    test_path = config["process_data_path"] + "test.csv"


if arg_output:
    output_path = arg_output
else:
    output_path = config["model_path"]

if arg_log_path:
    log_file = config["log_file"]
else:
    log_file = None

if arg_log_level:
    log_level = arg_log_level
else:
    log_level = "DEBUG"


logger = configure_logger(
    cfg=config["logging_default_config"],
    log_file=log_file,
    console=arg_no_console_log,
    log_level=log_level,
)
logger.info("Logging Start")

with mlflow.start_run():
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
    print("Save to: {}".format(mlflow.get_artifact_uri()))
