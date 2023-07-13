import argparse
import json
import logging
import logging.config
import os

from housing_project import housing_functions

with open("../config.json") as config_file:
    config = json.load(config_file)


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="INFO"
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """

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
    "-p", "--path", help="Specify the path to download the housing data"
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
arg_path = args.path
arg_log_path = args.log_path
arg_log_level = args.log_level
arg_no_console_log = args.no_console_log


if arg_path:
    HOUSING_PATH = arg_path
else:
    HOUSING_PATH = config["raw_data_path"]

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

# To download the file and save it to the path
housing_functions.fetch_housing_data(HOUSING_PATH)
logger.info("Housing Dataset Download complete")

# loading the dataset
housing = housing_functions.load_housing_data(HOUSING_PATH)
logger.info(f"Housing Dataset Loaded from {HOUSING_PATH}")

# Splitting the dataset to train and test
test_size = config["test_size"]
housing_train, housing_test = housing_functions.stratified_split_dataset(
    housing, test_size=test_size, random_state=42
)
logger.info(f"Test data split: {test_size}")

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
