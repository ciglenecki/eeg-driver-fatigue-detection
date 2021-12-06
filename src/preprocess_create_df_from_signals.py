from mne.epochs import Epochs
from preprocess_unzip_data import unzip_data
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

parser = argparse.ArgumentParser(description="Process arguments")
parser.add_argument("--users", metavar="N", type=int, help="Number of users that will be used (<= 12)")
parser.add_argument("--electrodes", metavar="N", type=int, nargs=2, help="Range of electrodes that will be used (N <= 30)")
parser.add_argument("--sig", metavar="N", type=int, help="Duration of signal (<= 300)")
parser.add_argument("--epoch-elems", metavar="N", type=int, help="Duration of signal (<= 1000)")
parser.add_argument("--unzip", metavar="N", type=bool, help="True/False")
parser.add_argument("--df-checkpoint", metavar="df", type=str, help="Precaculated entropy dataframe (the one that isn't cleaned and normalized)")
args = parser.parse_args()


is_full_iter = not any([args.users, args.electrodes, args.sig, args.epoch_elems])
user_count = args.users if (args.users) else USER_COUNT
signal_requested_seconds = args.sig if (args.sig) else SIGNAL_DURATION_SECONDS
if args.electrodes:
    subset = elect_cols[args.electrodes[0] : args.electrodes[1]]
    elect_cols_ignore = [electrode for electrode in elect_cols if electrode not in subset]
    elect_cols = subset

epoch_elems = args.epoch_elems if args.epoch_elems else FREQ

pickle_metadata = {
    "is_full_iter": is_full_iter,
    "user_count": user_count,
    "sig_seconds": signal_requested_seconds,
    "electrodes_ignored": elect_cols_ignore,
    "epoch_elems": epoch_elems,
}


# {(0,normal), (0,fatigue), (1,normal)...(12,fatigue)}
user_state_pairs = [(i_user, state) for i_user in range(0, user_count) for state in [NORMAL_STR, FATIGUE_STR]]
entropy_electrode_combinations = ["{}_{}".format(entropy, electrode) for entropy in ENTROPIES for electrode in elect_cols]


if args.unzip:
    unzip_data()


def get_epochs_from_signal(filename: str):
    eeg = read_raw_cnt(
        filename,
        eog=["HEOL", "HEOR", "VEOU", "VEOL"],
        preload=True,
        verbose=False,
    )
    eeg_filtered = eeg.notch_filter(50).filter(l_freq=0.15, h_freq=40)
    signal_seconds_floored = floor(len(eeg_filtered) / FREQ)
    tmin = signal_seconds_floored - signal_requested_seconds - SAFETY_CUTOFF_SECONDS
    tmax = signal_seconds_floored - SAFETY_CUTOFF_SECONDS
    eeg_filtered = eeg_filtered.crop(tmin=tmin, tmax=tmax)
    return make_fixed_length_epochs(eeg, duration=EPOCH_SECONDS, preload=False, verbose=False)


def get_df_from_epochs(epochs: Epochs, state: str):
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1, mag=1, grad=1))
    df = df.drop(["time", "condition", *elect_cols_ignore], axis=1)
    return df


def normalize_df(df: DataFrame, columns_to_scale: list):
    # set to 0 if treshold is met
    # NaN entropies can be set to zero
    # standard scaler scales for each column independently
    threshold = 1e-6
    df[df < threshold] = 0
    df[df <= 0] = 0
    df = df.fillna(0)
    df[columns_to_scale] = min_max_scaler(df[columns_to_scale])
    return df


if args.df_checkpoint:
    df = normalize_df(read_pickle(Path(args.df_checkpoint)), entropy_electrode_combinations)
    save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, "normalized")
    print("Only cleaning of existing df was performed.")
    sys.exit(1)

arr_users = []
for pair in user_state_pairs:
    print(pair)
    i_user, state = pair

    filename_user_signal = str(Path(PATH_DATASET_CNT, get_cnt_filename(i_user + 1, state)))
    epochs = get_epochs_from_signal(filename_user_signal)
    df = get_df_from_epochs(epochs, state)
    label = 1 if state == FATIGUE_STR else 0

    for i_poch in range(0, signal_requested_seconds):
        # filter rows for current epoch

        df_dict = {}
        df_epoch = df.loc[df["epoch"] == i_poch].head(epoch_elems)
        df_electrodes = df_epoch[elect_cols]

        # calculate all entropies for all electrodes
        df_spectral_entropy = df_electrodes.apply(func=lambda x: pd_spectral_entropy(x, freq=FREQ, standardize_input=True), axis=0)
        df_approximate_entropy = df_electrodes.apply(func=lambda x: pd_approximate_entropy(x, standardize_input=True), axis=0)
        df_sample_entropy = df_electrodes.apply(func=lambda x: pd_sample_entropy(x, standardize_input=True), axis=0)
        df_fuzzy_entropy = df_electrodes.apply(func=lambda x: pd_fuzzy_entropy(x, standardize_input=True), axis=0)

        # store entropies in a dictionary so they can be ordered correctly by using the ENTROPIES array
        df_dict = {
            "PE": df_spectral_entropy,
            "AE": df_approximate_entropy,
            "SE": df_sample_entropy,
            "FE": df_fuzzy_entropy,
        }

        # return list that contains the label and properly ordered entropies
        # [0, PE_FP1, PE_FP2, ... , PE_C3, AE_FP1, AE_FP2, ..., FE_C3]
        arr_users.append(
            [
                label,
                *df_dict[ENTROPIES[0]],
                *df_dict[ENTROPIES[1]],
                *df_dict[ENTROPIES[2]],
                *df_dict[ENTROPIES[3]],
            ]
        )


columns = ["label"] + entropy_electrode_combinations
df = DataFrame(arr_users, columns=columns)
df["label"] = df["label"].astype(int)

# save both raw and normalized
save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, "raw")
df = normalize_df(df, entropy_electrode_combinations)
glimpse_df(df)
save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, "normalized")
