import argparse

from pandas import DataFrame, set_option, read_pickle
from pathlib import Path
import warnings
import sys

from utils_file_saver import save_to_file_with_metadata

from pandas import DataFrame
import numpy as np
from utils_functions import min_max_dataframe, min_max_scaler_1d, standard_scale_dataframe
from utils_paths import PATH_DATAFRAME
from utils_env import training_columns_regex, NUM_USERS


def normalize_df(df: DataFrame, columns_to_scale: list, scaler=min_max_dataframe, per_person_scale=False):
    """
    Normalizes dataframe by replacing values and scaling them.
    Standard scaler scales for each column independently.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    # df[columns_to_scale] = df[columns_to_scale].apply(lambda x: mstats.winsorize(x, limits=[0.01, 0.01]), axis=0)
    if per_person_scale:
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

    output_dir = args.output_dir
    df = read_pickle(Path(args.df))
    training_column_names = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
    df = normalize_df(df, training_column_names, scaler=min_max_dataframe, per_person_scale=True)

    file_saver = lambda df, filename: df.to_pickle(str(filename))
    basename = Path(args.df).stem.replace("raw", "normalized-after")

    save_to_file_with_metadata(df, output_dir, basename, ".pkl", file_saver, metadata={})
    sys.exit(1)
