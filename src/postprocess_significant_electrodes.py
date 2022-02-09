"""
Finds significant electrodes by caculating weight described by the formula in the paper

Load already trained model via --svm argument (use it's hyperparameters)
Load the dataset with the --df argument
Refit the model (using the same hyperparameters) with a new train_test_split, because it's not known which data was used during the training phase 
Caculate weight for every channel and sort the list
Create a report file
"""

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
from utils_functions import get_timestamp, stdout_to_file
from utils_paths import PATH_REPORT
from utils_env import channels_good, NUM_USERS
from itertools import combinations
import sys
from datetime import datetime


set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("--svm", metavar="file", required=True, type=str, help="SVM model used for caclulating the accuracy")
parser.add_argument("--mode", metavar="users/all", required=True, type=str, choices=["users", "all"], help='Defines mode for caculating significant electrodes. "users" caculates weights for each user and then averages it. "all" uses all users at once.')
parser.add_argument("-r", "--output-report", metavar="dir", required=False, type=str, help="Directory where report file will be created.", default=PATH_REPORT)
args = parser.parse_args()
stdout_to_file(Path(args.output_report, "-".join(["significant-electrodes", args.mode, get_timestamp()]) + ".txt"))


df = read_pickle(args.df)
X = df.loc[:, ~df.columns.isin(["is_fatigued"])]
y = df.loc[:, df.columns.isin(["is_fatigued", "user_id"])]
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.5, random_state=0)
model: SVC = load_model(args.svm).best_estimator_

result = []
if args.mode == "users":
    result = caculate_mode_users(model, X_train_org, X_test_org, y_train_org, y_test_org, channels_good, 1)
else:
    result = caculate_mode_all(model, X_train_org, X_test_org, y_train_org, y_test_org, channels_good)

for line in result:
    print(line)
