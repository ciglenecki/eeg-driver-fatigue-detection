"""
Summary:
    Finds significant electrodes by caculating weight described by the formula in the paper

Steps:
    Load already trained model via --svm argument (use it's hyperparameters)
    Load the dataset with the --df argument
    Refit the model (using the same hyperparameters) with a new train_test_split, because it's not known which data was used during the training phase 
    Caculate weight for every channel and sort the list
    Create a report file

--------------------

Some electrodes are more important for predicting driver's state. To find the most significant electrodes we will use methods described in paper where we caculate weights for each electrode. Electrode with the highest weight is considered to be the most important.
"""

import argparse
from pathlib import Path

from pandas import DataFrame, read_pickle
from pandas._config.config import set_option
from postprocess_significant_electrodes_users import caculate_mode_drivers
from sklearn.svm import SVC

from postprocess_significant_electrodes_all import caculate_mode_all
from preprocess_preprocess_df import split_and_normalize, training_columns_regex
from utils_env import channels_good
from utils_file_saver import load_model
from utils_functions import get_timestamp, stdout_to_file
from utils_paths import PATH_REPORT

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()

parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("--svm", metavar="file", required=True, type=str, help="SVM model used for caclulating the accuracy")
parser.add_argument("--mode", metavar="drivers/all", required=True, type=str, choices=["drivers", "all"], help='Defines mode for caculating significant electrodes. "drivers" caculates weights for each driver and then averages it. "all" uses all drivers at once.')
parser.add_argument("-r", "--output-report", metavar="dir", required=False, type=str, help="Directory where report file will be created.", default=PATH_REPORT)
parser.add_argument(
    "--channels-ignore", nargs="+", help="List of channels (electrodes) that will be ignored. Possible values: [HEOL, HEOR, FP1, FP2, VEOU, VEOL, F7, F3, FZ, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, CZ, C4, T4, TP7, CP3, CPZ, CP4, TP8, A1, T5, P3, PZ, P4, T6, A2, O1, OZ, O2, FT9, FT10, PO1, PO2]"
)

parser.set_defaults(channels_ignore=[])
args = parser.parse_args()

stdout_to_file(Path(args.output_report, "-".join(["significant-electrodes", args.mode, get_timestamp()]) + ".txt"))

channels_ignore = args.channels_ignore
channels = list(set(channels_good) - set(channels_ignore))

df: DataFrame = read_pickle(args.df)
training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
X = df.loc[:, ~df.columns.isin(["is_fatigued"])]
y = df.loc[:, df.columns.isin(["is_fatigued", "driver_id"])]
X_train_org, X_test_org, y_train_org, y_test_org = split_and_normalize(X, y, training_columns, test_size=0.5)

model: SVC = load_model(args.svm).best_estimator_

result = []
if args.mode == "drivers":
    result = caculate_mode_drivers(model, X_train_org, X_test_org, y_train_org, y_test_org, channels, 1)
else:
    result = caculate_mode_all(model, X_train_org, X_test_org, y_train_org, y_test_org, channels)

for line in result:
    print(line)
