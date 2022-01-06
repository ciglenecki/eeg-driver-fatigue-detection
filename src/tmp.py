import argparse
from itertools import chain, combinations
from pathlib import Path
import sys
import warnings
from IPython.core.display import display
import numpy as np
from pandas import read_pickle
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
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
from utils_paths import PATH_MODEL, PATH_DATASET_MAT
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from os import getcwd
import scipy.io
import pandas as pd
from utils_functions import min_max_scaler
from time import sleep

"""
Temporary file used only for testing things around.
"""


set_option("display.max_columns", None)
a = np.zeros(shape=(3, 4, 10), dtype=[("a", np.int_), ("b", np.int_), ("c", np.int_)])


a["a"] = [[[1] * 10] * 4, [[2] * 10] * 4, [[3] * 10] * 4]

a["b"] = [[[1] * 10, [2] * 10, [3] * 10, [4] * 10]] * 3
a["c"] = [[[8] * 10] * 4] * 3
print(a)
print(a.shape)
# print(a["a"].reshape(3 * 4 * 10, 1))
one = a["a"].reshape(3 * 4 * 10, 1)
two = a["b"].reshape(3 * 4 * 10, 1)
three = a["c"].reshape(3 * 4, 10)
x1 = np.unique(np.concatenate((one, two), axis=1), axis=0)
print(x1)
nus = np.concatenate((x1, three), axis=1)
print(nus)
print()
print()
# print(a)
print(
    DataFrame.from_records(
        nus,
        columns=["a", "b", "c"],
    )
)
exit(1)
channels_good = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]
additional_feature_names = ["psd", "mean", "std"]
entropy_names = ["PE", "AE", "SE", "FE"]
feature_names = entropy_names + additional_feature_names


# (len(states), num_users, signal_duration, len(feature_names), len(channels_good)
states = 2
num_users = 10
sig = 300
channels_good
dtype = [("label", np.int_), ("user_id", np.int_), ("epoch", np.int_), ("feature", np.unicode_, 20), ("channel", np.unicode_, 20)]

# npy_matrix = np.array(
#     [[[list(product(channels_good, feature_names))] * sig] * num_users] * 2,
# )
npy_matrix = np.array([[[list(product(channels_good, feature_names))] * sig] * num_users] * 2, dtype=dtype)
print(npy_matrix)
print(npy_matrix.shape)
print(np.where(npy_matrix["channel"] == "b"))
exit(1)
mat2 = np.zeros((2, 2, 3), dtype=[("first", np.unicode_, 20), ("second", np.int_), ("third", np.int_)])
mat2["second"] = 2
print(np.where(mat2["first"] == 2))
print(mat2["first"] == "test")

print()
exit(1)

channel = np.array(channels_good)
feature = np.array(feature_names)

print(list(product(feature, [channels_good])))
print(dict.fromkeys(feature_names, channels_good))
arr = np.array(list(dict.fromkeys(feature_names, channels_good).items()))

dtype_fet = list(map(lambda x: (x, np.unicode_, 20), feature_names))
arr = np.full((7, 30), channel, dtype=dtype_fet)

print(arr.shape)

print(npy_matrix["feature"].shape)
print(npy_matrix["channel"].shape)

npy_matrix["feature"] = arr

print(npy_matrix)
# ones = np.ones((2, 3, 4, 7, 30), dtype=np.int_)
# print(ones)

# np.put_along_axis(npy_matrix, ones, channels_good, 4)

# print(npy_matrix[0, 0, 0, 0, 0])
# npy_matrix[:, :, :, :].fill(channels_good)

# df = load_dataframe("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes/complete-normalized-with_user_id-2021-12-08-16-10-19-.pkl")
# # df.drop("user_id")
# print(df["SE_P3"].describe())

# sys.stdin.close()
# print("test")
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
