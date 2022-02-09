import argparse
import pickle
import sys
import warnings
from itertools import chain, combinations, product
from os import getcwd
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import scipy.io
from IPython.core.display import display
from joblib import dump, load
from pandas import read_pickle
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
from scipy import stats
from scipy.stats import mstats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from preprocess_normalize_df import normalize_df
from utils_env import PAPER_BF_HIDDEN, PAPER_C, PAPER_G, PAPER_RFC_INPUT_VARIABLES, PAPER_RFC_TREES, channels_good, feature_names, training_columns_regex
from utils_file_saver import load_dataframe, save_df_to_disk, save_model, save_to_file_with_metadata
from utils_functions import glimpse_df, min_max_dataframe, min_max_scaler, powerset
from utils_paths import PATH_DATASET_MAT, PATH_MODEL

"""
Temporary file used only for testing things around.
"""


def get_column_names(use_brainbands: bool, brainwave_bands: dict):
    prod = product()
    if use_brainbands:
        prod = product(channels_good, feature_names, brainwave_bands.keys())
    else:
        prod = product(channels_good, feature_names)
    return list(map(lambda x: "_".join(x), prod))


set_option("display.max_columns", None)

df = load_dataframe("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes/complete-normalized-2022-01-10-00-40-53-is_complete_dataset=true__brains=false__ica=false__reref=true.pkl")

df2 = load_dataframe("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes/complete-normalized-2022-01-08-14-53-51__brains=false__ica=false.pkl")

print(df.head(n=1)["T5_AE"])
print(df2.head(n=1)["T5_AE"])

exit(1)
# df_new = load_dataframe("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes/complete-normalized-2022-01-07-09-33-21-.pkl")


# t3 = df.loc[:, df.columns.str.contains("T3")]

# print(t3.describe())

# df = normalize_df(df, get_column_names(False, {}))

# t3 = df.loc[:, df.columns.str.contains("T3")]
# print("Normalized")

# print(t3.describe())
# print("Old")
# print(df_new.loc[:, df_new.columns.str.contains("T3")].describe())

# basename = "a-complete-normalized-2022-01-07-09-33-20-"
# dir = "/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes"

# file_saver = lambda df, filename: df.to_pickle(str(filename))
# save_to_file_with_metadata(df, dir, basename, ".pkl", file_saver, metadata={})

# exit(1)
# df_org = df_org.replace([np.inf, -np.inf], np.nan)
# # df_org[df <= 0] = 0
# df_org = df_org.fillna(0)

# df = min_max_dataframe(df_org)

# print("Org", df["P4_mean"].describe())

# df = min_max_dataframe(df_org.apply(using_mstats, axis=0))
# print("ms", df["P4_mean"].describe())

# df = df[(np.abs(stats.zscore(df)) < 2)]
# print(df.shape[1] - sum(df.isnull().any()))
# df[:] = min_max_scaler(df[:])

# df[:] = min_max_scaler(df[:])

# print(df)


# print(df.describe())


# # df.drop("user_id")
# print(df["SE_P3"].describe())


a = np.zeros(shape=(3, 4, 10), dtype=[("a", np.int_), ("b", np.int_), ("c", np.int_)])


a["a"] = [[[1] * 10] * 4, [[2] * 10] * 4, [[3] * 10] * 4]

a["b"] = [[[1] * 10, [2] * 10, [3] * 10, [4] * 10]] * 3
a["c"] = [[[8] * 10] * 4] * 3
print(a)
print(a.shape)
# print(a["a"].reshape(3 * 4 * 10, 1))
one = a["a"].reshape(-1, 1)
print(a["a"].reshape(-1, 1))
two = a["b"].reshape(-1, 1)
three = a["c"].reshape(3 * 4, 10)
x1 = np.unique(np.concatenate((one, two), axis=1), axis=0)
print(x1)
nus = np.concatenate((x1, three), axis=1)
print(nus)
print()
print()
# print(a)
