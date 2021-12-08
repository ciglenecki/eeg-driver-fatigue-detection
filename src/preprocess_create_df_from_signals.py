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
from utils_file_saver import save_df_to_disk
from utils_paths import *
from utils_env import *
from utils_functions import *
from utils_entropy import *

set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--users", metavar="N", type=int, help="Number of users that will be used (1 >= N <= 12)")
parser.add_argument("--sig", metavar="N", type=int, help="Duration of the signal in seconds (1 >= N <= 300)")
parser.add_argument("--epoch-elems", metavar="N", type=int, help="Reduce (cut off) epoch duration to N miliseconds (11 >= N <= 1000)")
parser.add_argument("--df-checkpoint", metavar="df", type=str, help="Load precaculated entropy dataframe (the one that isn't cleaned and normalized)")
args = parser.parse_args()

is_complete_train = not any([args.users, args.sig, args.epoch_elems])
if is_complete_train:
    print("Performing complete training")
num_users = args.users if (args.users) else num_users
signal_duration = args.sig if (args.sig) else SIGNAL_DURATION_SECONDS_DEFAULT
epoch_elems = args.epoch_elems if args.epoch_elems else FREQ

pickle_metadata = {
    "is_complete_train": is_complete_train,
    "num_users": num_users,
    "sig_seconds": signal_duration,
    "epoch_elems": epoch_elems,
}


entropy_electrode_combinations = ["{}_{}".format(entropy, electrode) for entropy in entropy_names for electrode in elect_good]


def signal_to_epochs(filename: str):
    """
    Load the signal
    Exclude bad channels
    Crops the filter the signal
    Return epoches

    Notes:
    eeg = read_raw_cnt(filename, eog=["HEOL", "HEOR", "VEOU", "VEOL"], preload=True, verbose=False

    when comparing with and without eog, it changed data dramatically? this is what allowed me to sync data with S
    """

    eeg = read_raw_cnt(filename, preload=True, verbose=False)
    eeg.info["bads"].extend(elect_bad)
    eeg.pick_channels(elect_good)

    signal_total_duration = floor(len(eeg) / FREQ)
    start = signal_total_duration - signal_duration + signal_offset
    end = signal_total_duration + signal_offset
    eeg_filtered = eeg.crop(tmin=start, tmax=end).notch_filter(50).filter(l_freq=0.15, h_freq=40)

    return make_fixed_length_epochs(eeg_filtered, duration=EPOCH_SECONDS, preload=True, verbose=False)


def epochs_to_dataframe(epochs: Epochs):
    """
    Returns epochs converted to dataframe.
    Useless columns are excluded.
    """
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(["time", "condition", *elect_ignore, *elect_bad], axis=1)
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
    df = normalize_df(read_pickle(Path(args.df_checkpoint)), entropy_electrode_combinations)
    save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, "normalized")
    print("Only cleaning of existing df was performed.")
    sys.exit(1)

backup_matrix = np.zeros(shape=(len(states), num_users, len(entropy_names), signal_duration, len(elect_good)))
rows = []
for user_id in range(0, num_users):
    for state in states:
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
            df_electrodes = df_epoch[elect_good]

            df_spectral_entropy = df_electrodes.apply(func=lambda x: pd_spectral_entropy(x, freq=FREQ, standardize_input=True), axis=0)
            df_approximate_entropy = df_electrodes.apply(func=lambda x: pd_approximate_entropy(x, standardize_input=True), axis=0)
            df_sample_entropy = df_electrodes.apply(func=lambda x: pd_sample_entropy(x, standardize_input=True), axis=0)
            df_fuzzy_entropy = df_electrodes.apply(func=lambda x: pd_fuzzy_entropy(x, standardize_input=True), axis=0)

            df_dict = {
                "PE": df_spectral_entropy,
                "AE": df_approximate_entropy,
                "SE": df_sample_entropy,
                "FE": df_fuzzy_entropy,
            }

            for i, e in enumerate(entropy_names):
                backup_matrix[label][user_id][i][epoch_id] = np.array(df_dict[entropy_names[i]])

            rows.append([label, *df_dict[entropy_names[0]], *df_dict[entropy_names[1]], *df_dict[entropy_names[2]], *df_dict[entropy_names[3]]])

        if is_complete_train:
            np.save(str(Path(PATH_DATA_DATAFRAME, "_raw_matrix")), backup_matrix)
"""
Create dataframe from rows and columns
"""
columns = ["label"] + entropy_electrode_combinations
df = DataFrame(rows, columns=columns)
df["label"] = df["label"].astype(int)

"""
Complete training - save instantly so no error with naming is possible
"""
if is_complete_train:
    np.save(str(Path(PATH_DATA_DATAFRAME, "_raw_matrix")), backup_matrix)
    df.to_pickle(str(Path(PATH_DATA_DATAFRAME, "_raw")))


prefix = ""
if is_complete_train:
    pickle_metadata = {}
    prefix = "complete-"
else:
    prefix = "partial-"

np.save(str(Path(PATH_DATA_DATAFRAME, "raw_matrix" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S"))), backup_matrix)


save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, prefix + "raw")
df = normalize_df(df, entropy_electrode_combinations)
glimpse_df(df)

save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, prefix + "normalized")
