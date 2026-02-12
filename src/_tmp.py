"""
=== IGNORE THIS FILE ===
File used for testing things out
"""

import argparse
import random
import sys
import warnings
from itertools import product
from pathlib import Path
from turtle import title

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, read_pickle, set_option
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils_env import training_columns_regex
from utils_file_saver import get_decorated_filepath, save_figure, save_obj
from utils_paths import PATH_DATAFRAME


def split_and_normalize(X: Series, y: Series, test_size: float, columns_to_scale, scaler: MinMaxScaler = MinMaxScaler()):
    """Columns to scale can be both string list or list of bools"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train.loc[:, columns_to_scale])
    X_test.loc[:, columns_to_scale] = scaler.transform(X_test.loc[:, columns_to_scale])
    return X_train, X_test, y_train, y_test


def df_replace_values(df: DataFrame):
    """
    Normalizes dataframe by replacing values and scaling them.
    Standard scaler scales for each column independently.
    Scale per person
    """

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


if __name__ == "__main__":
    set_option("display.max_columns", None)
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--df", metavar="df", type=str, help="Load precaculated entropy dataframe (the one that isn't cleaned and normalized)")
    parser.add_argument("--output-dir", metavar="dir", type=str, help="Directory where dataframe and npy files will be saved", default=PATH_DATAFRAME)
    args = parser.parse_args()

    df_path = Path(args.df)
    output_dir = Path(args.output_dir)

    df: DataFrame = read_pickle(df_path)
    training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)

    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    normal_df = (df - df.min()) / (df.max() - df.min())
    print("normal df", normal_df.shape)
    normal_df = normal_df[normal_df.columns[normal_df.max() != 0]]  # remove constant attributes
    df = normal_df.loc[:, normal_df.std() > 0.1]

    print("non constant normal df", df.shape)

    # clustering = DBSCAN(eps=1.75, min_samples=2).fit(normal_df)
    # df = df.loc[clustering.labels_ != -1, :]
    # df = (df - df.min()) / (df.max() - df.min())
    # print(df.shape)
    print(df.columns)

    categories = np.unique(df["is_fatigued"])
    colors = np.linspace(0, 1, len(categories))
    print(colors)
    colordict = dict(zip(categories, colors))
    df["Color"] = df["is_fatigued"].apply(lambda x: colordict[x])
    print(df["Color"])

    for col_name, col in sorted(df.iteritems(), key=lambda k: random.random()):
        fig, ax = plt.subplots()
        df.loc[df["is_fatigued"] == 1, col_name].hist(color="red", ax=ax, alpha=0.3, bins=50)
        df.loc[df["is_fatigued"] == 0, col_name].hist(color="green", ax=ax, alpha=0.3, bins=50)
        # plt.hist(col, bins=1000, color=df.Color.values + 100)
        # axis[1].hist(normal_df.loc[:, col_name], bins=1000)
        plt.title(label=col_name)
        plt.show()
        # save_figure(Path("/home/matej/2-fer/uuzop/eeg-driver-fatigue-detection/fig.png"), {})
        print(col_name, np.std(col))
# update

