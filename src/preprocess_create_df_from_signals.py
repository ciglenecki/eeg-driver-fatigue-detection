import mne
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

parser = argparse.ArgumentParser()
parser.add_argument("--users", metavar="N", type=int, help="Number of users that will be used (1 >= N <= 12)")
parser.add_argument("--sig", metavar="N", type=int, help="Duration of the signal in seconds (1 >= N <= 300)")
parser.add_argument("--epoch-elems", metavar="N", type=int, help="Reduce (cut off) epoch duration to N miliseconds (1 >= N <= 1000)")
parser.add_argument("--unzip", metavar="N", type=bool, help="Unzip the dataset from zips (True/False)")
parser.add_argument("--df-checkpoint", metavar="df", type=str, help="Load precaculated entropy dataframe (the one that isn't cleaned and normalized)")
args = parser.parse_args()

is_complete_train = not any([args.users, args.sig, args.epoch_elems])
user_count = args.users if (args.users) else USER_COUNT
signal_duration_target_s = args.sig if (args.sig) else SIGNAL_DURATION_SECONDS
epoch_elems = args.epoch_elems if args.epoch_elems else FREQ

pickle_metadata = {
    "is_complete_train": is_complete_train,
    "user_count": user_count,
    "sig_seconds": signal_duration_target_s,
    "epoch_elems": epoch_elems,
}


# {(0,normal), (0,fatigue), (1,normal)...(12,fatigue)}
user_state_pairs = [(i_user, state) for i_user in range(0, user_count) for state in [NORMAL_STR, FATIGUE_STR]]
entropy_electrode_combinations = ["{}_{}".format(entropy, electrode) for entropy in ENTROPIES for electrode in elect_good]


if args.unzip:
    unzip_data()


def get_epochs_from_signal(filename: str):

    # eeg = read_raw_cnt(filename, eog=["HEOL", "HEOR", "VEOU", "VEOL"], preload=True, verbose=False)
    # when comparing with and without eog, it changed data dramatically? this is what allowed me to sync data with S
    eeg = read_raw_cnt(filename, preload=True, verbose=False)

    # Exclude bad channels
    eeg.info["bads"].extend(elect_bad)
    eeg.pick_channels(elect_good)

    # Crop and filter the data
    signal_duration_s = floor(len(eeg) / FREQ)
    tmin = signal_duration_s - signal_duration_target_s + signal_offset_s
    tmax = signal_duration_s + signal_offset_s
    eeg_filtered = eeg.crop(tmin=tmin, tmax=tmax).notch_filter(50).filter(l_freq=0.15, h_freq=40)

    return make_fixed_length_epochs(eeg_filtered, duration=EPOCH_SECONDS, preload=True, verbose=False)


def get_df_from_epochs(epochs: Epochs):
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(["time", "condition", *elect_ignore], axis=1)
    return df


def normalize_df(df: DataFrame, columns_to_scale: list):
    # set to 0 if treshold is met
    # NaN entropies can be set to zero
    # standard scaler scales for each column independently

    # threshold = 1e-6
    # df[df < threshold] = 0
    df[df <= 0] = 0
    df = df.fillna(0)
    df[columns_to_scale] = min_max_scaler(df[columns_to_scale])
    return df


if args.df_checkpoint:
    df = normalize_df(read_pickle(Path(args.df_checkpoint)), entropy_electrode_combinations)
    save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, "normalized")
    print("Only cleaning of existing df was performed.")
    sys.exit(1)

rows = []
for pair in user_state_pairs:
    print(pair)
    i_user, state = pair

    filename_user_signal = str(Path(PATH_DATASET_CNT, get_cnt_filename(i_user + 1, state)))
    epochs = get_epochs_from_signal(filename_user_signal)
    df = get_df_from_epochs(epochs)

    label = 1 if state == FATIGUE_STR else 0

    for i_poch in range(0, signal_duration_target_s):
        # filter rows for current epoch
        df_dict = {}
        df_epoch = df.loc[df["epoch"] == i_poch].head(epoch_elems)
        df_electrodes = df_epoch[elect_good]

        # calculate all entropies for all electrodes
        df_spectral_entropy = df_electrodes.apply(func=lambda x: pd_spectral_entropy(x, freq=FREQ, standardize_input=False), axis=0)
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
        rows.append(
            [
                label,
                *df_dict[ENTROPIES[0]],
                *df_dict[ENTROPIES[1]],
                *df_dict[ENTROPIES[2]],
                *df_dict[ENTROPIES[3]],
            ]
        )


columns = ["label"] + entropy_electrode_combinations
df = DataFrame(rows, columns=columns)
df["label"] = df["label"].astype(int)

# save both raw and normalized
file_path = Path(PATH_DATA_DATAFRAME, "raw")
df.to_pickle(str(file_path))

save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, "raw")
df = normalize_df(df, entropy_electrode_combinations)
glimpse_df(df)
save_df_to_disk(df, pickle_metadata, PATH_DATA_DATAFRAME, "normalized")
