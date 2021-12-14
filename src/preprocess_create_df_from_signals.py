"""
Creates a tabluar dataframe file which is used for training with the following form:

| user_id | epoch_id | label | PE_CH01 | PE_CH02 | ... | PE_CH30 | SE_CH01 | SE_CH02 | ... | FE_CH30 |
| ------- | -------- | ----- | ------- | ------- | --- | ------- | ------- | ------- | --- | ------- |
| 01      | 0        | 0     | 0.3     | 0.23    | ... | 0.6     | 0.8     | 0.1     | ... | 0.2     |
| 01      | 1        | 0     | 0.2     | 0.1     | ... | 0       | 0.2     | 0.1     | ... | 0.2     |
| ...     | ...      | ...   | ...     | ...     | ... | ...     | ...     | ...     | ... | ...     |
| 01      | 0        | 0     | 0.6     | 0.3     | ... | 0.1     | 0.2     | 0.5     | ... | 0.1     |
| 02      | 1        | 0     | 0.2     | 0.1     | ... | 0       | 0.2     | 0.1     | ... | 0.2     |
| ...     | ...      | ...   | ...     | ...     | ... | ...     | ...     | ...     | ... | ...     |

Number of rows: users (12) * epochs (300) * states (2) = 7200
Number of columns: user_id (1) + label (1) + epoch_id (1) + entropies (4) * channels (30) = 123

The dataframe file is saved at ./data/dataframes by default with name
File with prefix "complete-normalized" should be used for training later on. 
"""

from datetime import datetime
from mne.epochs import Epochs
from mne import make_fixed_length_epochs
import argparse
from math import floor
from mne.io import read_raw_cnt
from pandas import DataFrame, set_option, read_pickle
from pathlib import Path
import warnings
import sys
from tqdm import tqdm
from utils_file_saver import save_df_to_disk, save_npy_to_disk
from utils_paths import *
from utils_env import *
from utils_functions import *
from utils_entropy import *

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--users", metavar="N", type=int, help="Number of users that will be used (1 >= N <= 12)")
parser.add_argument("--sig", metavar="N", type=int, help="Duration of the signal in seconds (1 >= N <= 300)")
parser.add_argument("--epoch-elems", metavar="N", type=int, help="Reduce (cut off) epoch duration to N miliseconds (11 >= N <= 1000)")
parser.add_argument("--df-checkpoint", metavar="df", type=str, help="Load precaculated entropy dataframe (the one that isn't cleaned and normalized)")
parser.add_argument("--output-dir", metavar="dir", type=str, help="Directory where dataframe and npy files will be saved", default=PATH_DATAFRAME)
args = parser.parse_args()

is_complete_train = not any([args.users, args.sig, args.epoch_elems])
is_complete_train = True
print("Training on {} dataset...".format("complete" if is_complete_train else "partial"))
num_users = args.users if (args.users) else num_users
signal_duration = args.sig if (args.sig) else SIGNAL_DURATION_SECONDS_DEFAULT
epoch_elems = args.epoch_elems if args.epoch_elems else FREQ
output_dir = args.output_dir
train_metadata = {
    "is_complete_train": is_complete_train,
    "num_users": num_users,
    "sig_seconds": signal_duration,
    "epoch_elems": epoch_elems,
}


def signal_to_epochs(filename: str):
    """
    Load the signal.
    Exclude bad channels.
    Crops the filter the signal.
    Return epoches.

    Notes:
    eeg = read_raw_cnt(filename, eog=["HEOL", "HEOR", "VEOU", "VEOL"], preload=True, verbose=False
    when comparing with and without eog, it changed data dramatically? this is what allowed me to sync data with S
    """
    eeg = read_raw_cnt(filename, preload=True, verbose=False)
    eeg.info["bads"].extend(channels_bad)
    eeg.pick_channels(channels_good)

    signal_total_duration = floor(len(eeg) / FREQ)
    start = signal_total_duration - signal_duration + signal_offset
    end = signal_total_duration + signal_offset
    low_freq, high_freq = LOW_PASS_FILTER_RANGE_HZ
    eeg_filtered = eeg.crop(tmin=start, tmax=end).notch_filter(NOTCH_FILTER_HZ).filter(l_freq=low_freq, h_freq=high_freq)

    return make_fixed_length_epochs(eeg_filtered, duration=EPOCH_SECONDS, preload=True, verbose=False)


def epochs_to_dataframe(epochs: Epochs):
    """
    Returns epochs converted to dataframe.
    Useless columns are excluded.
    """
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(["time", "condition", *channels_ignore], axis=1)
    return df


def normalize_df(df: DataFrame, columns_to_scale: list):
    """
    Normalizes dataframe by replacing values and scaling them.
    Standard scaler scales for each column independently.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df[df <= 0] = 0
    df = df.fillna(0)
    df[columns_to_scale] = min_max_scaler(df[columns_to_scale])
    return df


if args.df_checkpoint:
    """
    If checkpoint only action to perform is normalizing since entropies are already caculated
    """
    df = normalize_df(read_pickle(Path(args.df_checkpoint)), entropy_channel_combinations)
    save_df_to_disk(df, train_metadata, output_dir, "normalized")
    print("Only cleaning of existing df was performed.")
    sys.exit(1)

npy_matrix = np.zeros(shape=(len(states), num_users, len(entropy_names), signal_duration, len(channels_good)))
rows = []
for user_id in tqdm(range(0, num_users)):
    for state in tqdm(states):
        file_signal = str(Path(PATH_DATASET_CNT, get_cnt_filename(user_id + 1, state)))
        epochs = signal_to_epochs(file_signal)
        df = epochs_to_dataframe(epochs)
        label = 1 if state == FATIGUE_STR else 0

        for epoch_id in range(0, signal_duration):
            """
            Filter dataframe rows that have the current epoch are selected.
            Caculate entropy array for all channels. Shape (30,)
            Create a simple dictionary and use to return append entropies in a proper.
            Order of entropies is defined by list entropy_names.

            Append to backup matrix
            Append to list that contains the label and properly ordered entropies
            e.g. [0, PE_FP1, PE_FP2, ... , PE_C3, AE_FP1, AE_FP2, ..., FE_C3]
            """

            df_dict = {}
            df_epoch = df.loc[df["epoch"] == epoch_id].head(epoch_elems)
            df_channels = df_epoch[channels_good]

            SE = df_channels.apply(func=lambda x: pd_spectral_entropy(x, freq=FREQ), axis=0)
            AE = df_channels.apply(func=lambda x: pd_approximate_entropy(x), axis=0)
            SE = df_channels.apply(func=lambda x: pd_sample_entropy(x), axis=0)
            FE = df_channels.apply(func=lambda x: pd_fuzzy_entropy(x), axis=0)

            df_dict = {
                "PE": SE,
                "AE": AE,
                "SE": SE,
                "FE": FE,
            }

            for i in range(len(entropy_names)):
                npy_matrix[label][user_id][i][epoch_id] = np.array(df_dict[entropy_names[i]])

            rows.append([label, user_id, epoch_id, *df_dict[entropy_names[0]], *df_dict[entropy_names[1]], *df_dict[entropy_names[2]], *df_dict[entropy_names[3]]])

        if is_complete_train:
            np.save(str(Path(output_dir, ".raw_matrix")), npy_matrix)

"""Create dataframe from rows and columns"""
columns = ["label", "user_id", "epoch_id"] + entropy_channel_combinations
df = DataFrame(rows, columns=columns)
df["label"] = df["label"].astype(int)

"""Complete training - save instantly so no error with naming is possible"""
if is_complete_train:
    np.save(str(Path(output_dir, ".raw_npy.npy")), npy_matrix)
    df.to_pickle(str(Path(output_dir, ".raw_df.pkl")))

"""Save to files"""
save_npy_to_disk(npy_matrix, output_dir, "npy_matrix", train_metadata)
save_df_to_disk(df, is_complete_train, output_dir, "raw-with-userid", train_metadata)
df = normalize_df(df, entropy_channel_combinations)
glimpse_df(df)
save_df_to_disk(df, is_complete_train, output_dir, "normalized-with-userid", train_metadata)
df = df.drop(["user_id", "epoch_id"], axis=1)
save_df_to_disk(df, is_complete_train, output_dir, "normalized", train_metadata)
