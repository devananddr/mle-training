import argparse
import json
import logging
import logging.config
import pickle

import pandas as pd

from housing_project import housing_functions

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
    "-im",
    "--input-model",
    help="Specify the path for model",
)
parser.add_argument(
    "-id",
    "--input-data",
    help="Specify the path for test dataset",
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
arg_input_model = args.input_model
arg_input_data = args.input_data
arg_log_path = args.log_path
arg_log_level = args.log_level
arg_no_console_log = args.no_console_log


if arg_input_model:
    model_path = arg_input_model

else:
    model_path = config["model_path"]


if arg_input_data:
    test_data_path = arg_input_data
else:
    test_data_path = config["process_data_path"] + "test.csv"

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

if log_file == None:
    logger.warning(
        "Logging results in log file is recommended. To save the logs, specify the argument from console"
    )

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
