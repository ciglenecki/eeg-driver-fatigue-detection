import argparse
import pickle
import sys
import warnings
from collections import Counter
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

from utils_env import training_columns_regex
from utils_file_saver import load_dataframe

"""
Temporary file used only for testing things around.
"""

def sum_dict(d1, d2):
	d1 = Counter(d1)
	d2 = Counter(d2)
	z = dict(Counter(x)+Counter(y))

dict1 = {"a": 1, "b": 2, "c": {"a": 1, "b": 2}}
dict2 = {"a": 1, "b": 2, "c": {"a": 5, "b": 5}}

print(Counter(dict1) + Counter(dict2))
print(dict1 + dict2)
# def get_column_names(use_brainbands: bool, brainwave_bands: dict):
#     prod = product()
#     if use_brainbands:
#         prod = product(channels_good, feature_names, brainwave_bands.keys())
#     else:
#         prod = product(channels_good, feature_names)
#     return list(map(lambda x: "_".join(x), prod))


# set_option("display.max_columns", None)

# df = load_dataframe("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/data/dataframes/partial-raw-2022-01-08-01-23-36-is_complete_train=false__brains=false__ica=true.pkl")

# training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)

# exit(1)
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


# # df.drop("driver_id")
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
