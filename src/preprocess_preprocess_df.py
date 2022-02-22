"""
Normalized the dataframe.
Features will be scaled to [0,1]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from pandas import DataFrame, Series, read_pickle, set_option
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils_env import training_columns_regex
from utils_file_saver import get_decorated_filepath, save_obj
from utils_paths import PATH_DATAFRAME


def split_and_normalize(X: Series, y: Series, test_size: float, columns_to_scale, scaler: MinMaxScaler = MinMaxScaler()):
    """Columns to scale can be both string list or list of bools"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    X_train: Series
    X_test: Series
    y_train: Series
    y_test: Series
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
    df = df_replace_values(df)

    basename = df_path.stem.replace("raw", "cleaned2")
    file_saver = lambda df, filepath: DataFrame.to_pickle(df, filepath)
    filepath = get_decorated_filepath(directory=output_dir, basename=basename, extension=".pkl")
    save_obj(obj=df, filepath=filepath, file_saver=DataFrame.to_pickle, metadata={})
