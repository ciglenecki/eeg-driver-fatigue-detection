import argparse
from itertools import chain, combinations
from math import gamma
from os import getcwd
from pathlib import Path
from typing import Dict, List
import warnings
from IPython.core.display import display
from pandas import read_pickle
from pandas._config.config import set_option
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from postprocess_significant_electrodes_all import caculate_mode_all
from postprocess_significant_electrodes_users import caculate_mode_users
from utils_file_saver import load_model
from utils_functions import stdout_to_file
from utils_paths import PATH_REPORT
from utils_env import channels_good, num_users
from itertools import combinations
import sys
from datetime import datetime

"""
Get the trained model.
Get the dataset
Since we don't know which data was used in training we will refit the model with same optimal parameters.
Caculate weight for every channel and sort by weight.
"""

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("--svm", metavar="file", required=True, type=str, help="SVM model used for caclulating the accuracy")
parser.add_argument("--mode", metavar="users/all", required=True, type=str, choices=["users", "all"], help='Defines mode for caculating significant electrodes. "users" caculates weights for each user and then averages it. "all" uses all users at once.')
args = parser.parse_args()
timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
report_filename = Path(PATH_REPORT, "-".join(["significant-electrodes", args.mode, timestamp]) + ".txt")
print(report_filename)
stdout_to_file(report_filename)
print("Results")
model: SVC = load_model(args.svm).best_estimator_

df = read_pickle(args.df)
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, df.columns.isin(["label", "user_id"])]

X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.5, random_state=0)

result = None
if args.mode == "users":
    result = caculate_mode_users(model, X_train_org, X_test_org, y_train_org, y_test_org, channels_good, 1)
else:
    result = caculate_mode_all(model, X_train_org, X_test_org, y_train_org, y_test_org, channels_good)
print(result)
sys.stdout.close()
