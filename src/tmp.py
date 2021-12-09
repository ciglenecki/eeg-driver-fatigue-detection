import argparse
from itertools import chain, combinations
from pathlib import Path
import sys
import warnings
from IPython.core.display import display
import numpy as np
from pandas import read_pickle
from pandas._config.config import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
from joblib import dump, load
from tqdm import tqdm
from utils_env import PAPER_BF_HIDDEN, PAPER_C, PAPER_G, PAPER_RFC_INPUT_VARIABLES, PAPER_RFC_TREES
from utils_file_saver import load_dataframe, save_df_to_disk, save_model
from sklearn.neural_network import MLPClassifier
from utils_functions import glimpse_df, min_max_dataframe, powerset
from utils_paths import PATH_DATA_MODEL, PATH_DATASET_MAT
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from os import getcwd
import scipy.io
import pandas as pd
from utils_functions import min_max_scaler
from time import sleep

sys.stdin.close()
print("test")
# parser = argparse.ArgumentParser()
# # parser.add_argument("--model", metavar="model", required=True, type=str, help="Model")
# args = parser.parse_args()


# print(np.load(Path(getcwd(), "TWO_ELECTRODE_SVM_ACCURACY_SCORE.npy"), allow_pickle=True))


# df = load_dataframe("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes/complete-normalized-2021-12-08-09-48-01-.pkl")

# new_f = "/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes/complete-normalized-with-userid-2021-12-08-16-08-01-.pkl"


# df["user_id"] = pd.Series(np.array([-1] * 7200))
# print(df["user_id"])
# # glimpse_df(df)

# # print(Path(getcwd(), args.model))

# for i, user_index in enumerate(list(range(0, 7200, 600))):
#     start = user_index
#     end = user_index + 600
#     print(start, end)
#     df["user_id"][start:end] = i

# save_df_to_disk(df, True, Path("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/datafram2es/"), "with_user_id", {})
# glimpse_df(df)

# print(df.tail(n=7200 - 599).head(n=30))
