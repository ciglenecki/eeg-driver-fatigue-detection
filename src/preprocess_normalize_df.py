"""
Normalized the dataframe.
Features will be scaled to [0,1]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from pandas import DataFrame, read_pickle, set_option

from utils_env import NUM_USERS, training_columns_regex
from utils_file_saver import save_to_file_with_metadata
from utils_functions import min_max_dataframe, min_max_scaler_1d, standard_scale_dataframe
from utils_paths import PATH_DATAFRAME


def normalize_df(df: DataFrame, columns_to_scale: list, scaler=min_max_dataframe, scale_per_person=False):
    """
    Normalizes dataframe by replacing values and scaling them.
    Standard scaler scales for each column independently.
    Scale per person
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    # df[columns_to_scale] = df[columns_to_scale].apply(lambda x: mstats.winsorize(x, limits=[0.01, 0.01]), axis=0)
    if scale_per_person:
        for i in range(NUM_USERS):
            df.loc[df["user_id"] == i, columns_to_scale] = scaler(df.loc[df["user_id"] == i, columns_to_scale])
    else:
        df[columns_to_scale] = scaler(df[columns_to_scale])
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
    training_column_names = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
    df = normalize_df(df, training_column_names, scaler=min_max_dataframe, per_person_scale=True)

    file_saver = lambda df, filename: df.to_pickle(str(filename))
    basename = df_path.stem.replace("raw", "normalized-after")
    save_to_file_with_metadata(df, output_dir, basename, ".pkl", file_saver, metadata={})
