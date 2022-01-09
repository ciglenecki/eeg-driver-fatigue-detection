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
from multiprocessing import Array
import mne
from mne.epochs import Epochs
from mne.preprocessing import ICA, regress_artifact
from mne import make_fixed_length_epochs, time_frequency
import argparse
from math import floor
from mne.io import read_raw_cnt
from mne.io.base import BaseRaw
from mne.transforms import scaling
from pandas import DataFrame, set_option, read_pickle
from pathlib import Path
import warnings
import sys
from scipy import signal
from tqdm import tqdm
from preprocess_normalize_df import normalize_df
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
from utils_file_saver import save_df_to_disk, save_npy_to_disk
from utils_paths import *
from utils_env import *
from utils_functions import *
from utils_feature_extraction import *
from itertools import product


set_option("display.max_columns", None)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--users", metavar="N", type=int, help="Number of users that will be used (1 >= N <= 12)")
parser.add_argument("--sig", metavar="N", type=int, help="Duration of the signal in seconds (1 >= N <= 300)")
parser.add_argument("--epoch-elems", metavar="N", type=int, help="Reduce (cut off) epoch duration to N miliseconds (11 >= N <= 1000)")
parser.add_argument("--df-checkpoint", metavar="df", type=str, help="Load precaculated entropy dataframe (the one that isn't cleaned and normalized)")
parser.add_argument("--output-dir", metavar="dir", type=str, help="Directory where dataframe and npy files will be saved", default=PATH_DATAFRAME)
parser.add_argument("--no-brainbands", dest="use_brainbands", action="store_false", help="Decompose signal into alpha and beta bands")
parser.add_argument("--use-ica", dest="use_ica", action="store_true", help="Apply ICA for each subject")
parser.add_argument("--use-reref", dest="use_reref", action="store_true", help="Apply channel rereferencing")

parser.set_defaults(brainbands=USE_BRAIN_BANDS)
parser.set_defaults(use_ica=USE_ICA)
parser.set_defaults(use_reref=USE_REREF)


args = parser.parse_args()

is_complete_train = not any([args.users, args.sig, args.epoch_elems])
print("Training on {} dataset...".format("complete" if is_complete_train else "partial"))
num_users = args.users if (args.users) else num_users
signal_duration = args.sig if (args.sig) else SIGNAL_DURATION_SECONDS_DEFAULT
epoch_elems = args.epoch_elems if args.epoch_elems else FREQ
output_dir = args.output_dir
use_brainbands = args.use_brainbands
use_ica = args.use_ica
use_reref = args.use_reref

train_metadata = {"is_complete_train": is_complete_train, "brains": args.use_brainbands, "ica": args.use_ica, "reref": args.use_reref}
brainwave_bands = get_brainwave_bands() if (use_brainbands) else {"placeholder": (0, 40)}


def signal_handle(filename: str):
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
    return eeg.crop(tmin=start, tmax=end).notch_filter(np.arange(NOTCH_FILTER_HZ, (NOTCH_FILTER_HZ * 5) + 1, NOTCH_FILTER_HZ)).filter(l_freq=low_freq, h_freq=high_freq)


def filter_brain_bandpass(signal: BaseRaw, brainwave_bands: Dict):
    result = {}
    for band_name, range_val in brainwave_bands.items():
        result[band_name] = signal.filter(l_freq=range_val[0], h_freq=range_val[1])
    return result


def signal_dict_to_epoch_dict(signal_dict: Dict[str, Epochs], EPOCH_SECONDS):
    result = {}
    for key, signal in signal_dict.items():
        result[key] = make_fixed_length_epochs(signal, duration=EPOCH_SECONDS, preload=True, verbose=False)
    return result


def epochs_to_dataframe(epochs: Epochs):
    """
    Returns epochs converted to dataframe.
    Useless columns are excluded.
    """
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(["time", "condition", *channels_ignore], axis=1)
    return df


def get_column_name(feature: str, channel: str, bandname: str = None):
    result = "_".join([channel, feature])
    if use_brainbands:
        return "_".join([result, bandname])
    return result


def get_column_names(use_brainbands: bool, brainwave_bands: dict):
    prod = product()
    if use_brainbands:
        prod = product(channels_good, feature_names, brainwave_bands.keys())
    else:
        prod = product(channels_good, feature_names)
    return list(map(lambda x: "_".join(x), prod))


training_cols = get_column_names(use_brainbands, brainwave_bands)
if args.df_checkpoint:
    """
    If checkpoint only action to perform is normalizing since features are already caculated
    """
    df = read_pickle(Path(args.df_checkpoint))
    training_column_names = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
    df = normalize_df(df, training_column_names)
    save_df_to_disk(df, train_metadata, output_dir, "normalized")
    print("Only cleaning of existing df was performed.")
    sys.exit(1)


df_dict = {k: [] for k in ["label", "user_id", "epoch_id", *training_cols]}

for user_id, state in tqdm(list(product(range(0, num_users), states))):
    label = 1 if state == FATIGUE_STR else 0
    file_signal = str(Path(PATH_DATASET_CNT, get_cnt_filename(user_id + 1, state)))
    signal_clean = signal_handle(file_signal)

    signal_brain_filtered = filter_brain_bandpass(signal_clean, brainwave_bands)
    epochs_brain_filtered = signal_dict_to_epoch_dict(signal_brain_filtered, EPOCH_SECONDS)

    for i_brainband, (bandname, epochs) in enumerate(tqdm(epochs_brain_filtered.items())):
        pdfs, _ = time_frequency.psd_welch(epochs, n_fft=FREQ, n_per_seg=FREQ, n_overlap=0, verbose=False)

        if use_reref:
            signal_clean.set_eeg_reference(ref_channels="average", ch_type="eeg")

        if use_ica:
            signal_l1 = signal_clean.copy().load_data().filter(l_freq=1, h_freq=None)
            ica = UnsupervisedSpatialFilter(FastICA(len(channels_good)), average=False)
            ica_data = ica.fit_transform(epochs.get_data())
            ica_data = ica_data.reshape(FREQ * signal_duration, len(channels_good))
            ica_data = np.insert(ica_data, 0, np.repeat(np.arange(0, signal_duration, 1), FREQ), axis=1)
            df = DataFrame.from_records(ica_data, columns=["epoch", *channels_good])
        else:
            df = epochs_to_dataframe(epochs)

        for epoch_id in tqdm(list(range(0, signal_duration))):
            """
            Filter dataframe rows that have the current epoch are selected.
            Caculate entropy array for all channels. Shape (30,)
            Create a simple dictionary and use to return append entropies in a proper.
            Order of entropies is defined by list entropy_names.

            Append to backup matrix
            Append to list that contains the label and properly ordered entropies
            e.g. [0, PE_FP1, PE_FP2, ... , PE_C3, AE_FP1, AE_FP2, ..., FE_C3]
            """

            df_epoch = df.loc[df["epoch"] == epoch_id].head(epoch_elems)
            df_channels = df_epoch[channels_good]

            lfreq = brainwave_bands[bandname][0]
            hfreq = brainwave_bands[bandname][1]

            mean = df_channels.apply(func=lambda x: np.mean(x), axis=0)
            std = df_channels.apply(func=lambda x: np.std(x), axis=0)
            pdf = np.apply_along_axis(lambda x: np.mean(x), axis=1, arr=pdfs[epoch_id, :, lfreq:hfreq])
            PE = df_channels.apply(func=lambda x: pd_spectral_entropy(x, freq=FREQ), axis=0)
            AE = df_channels.apply(func=lambda x: pd_approximate_entropy(x), axis=0)
            SE = df_channels.apply(func=lambda x: pd_sample_entropy(x), axis=0)
            FE = df_channels.apply(func=lambda x: pd_fuzzy_entropy(x), axis=0)

            for i_ch, channel in enumerate(channels_good):
                df_dict[get_column_name("mean", channel, bandname)].append(mean[i_ch])
                df_dict[get_column_name("std", channel, bandname)].append(std[i_ch])
                df_dict[get_column_name("pdf", channel, bandname)].append(pdf[i_ch])
                df_dict[get_column_name("PE", channel, bandname)].append(PE[i_ch])
                df_dict[get_column_name("AE", channel, bandname)].append(AE[i_ch])
                df_dict[get_column_name("SE", channel, bandname)].append(SE[i_ch])
                df_dict[get_column_name("FE", channel, bandname)].append(FE[i_ch])

            if i_brainband == 0:
                df_dict["user_id"].append(user_id)
                df_dict["label"].append(label)
                df_dict["epoch_id"].append(epoch_id)

"""Create dataframe from rows and columns"""
df = DataFrame.from_dict(df_dict)
df["label"] = df["label"].astype(int)
df["user_id"] = df["user_id"].astype(int)
df["epoch_id"] = df["epoch_id"].astype(int)

"""Complete training - save instantly so no error with naming is possible"""
if is_complete_train:
    # np.save(str(Path(output_dir, ".raw_npy.npy")), npy_matrix)
    df.to_pickle(str(Path(output_dir, ".raw_df.pkl")))

"""Save to files"""
# save_npy_to_disk(npy_matrix, output_dir, "npy_matrix", train_metadata)
save_df_to_disk(df, is_complete_train, output_dir, "raw", train_metadata)
df = normalize_df(df, training_cols)
glimpse_df(df)
save_df_to_disk(df, is_complete_train, output_dir, "normalized", train_metadata)
