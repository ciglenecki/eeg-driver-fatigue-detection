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

Number of rows: users (12) * epochs (300) * driving_states (2) = 7200
Number of columns: user_id (1) + label (1) + epoch_id (1) + entropies (4) * channels (30) = 123

The dataframe file is saved at ./data/dataframes by default with name
File with prefix "complete-normalized" should be used for training later on. 
"""

from typing import List
import argparse
from math import floor
from mne.io.cnt import read_raw_cnt
from mne.io.base import BaseRaw
from mne import Epochs
from mne.epochs import make_fixed_length_epochs
from pandas import DataFrame, set_option, read_pickle
from pathlib import Path
import warnings
import sys
from tqdm import tqdm
from preprocess_normalize_df import normalize_df
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import FastICA
from itertools import product
import numpy as np
from utils_functions import serialize_functions, get_cnt_filename, glimpse_df, is_arg_default
from utils_paths import PATH_DATAFRAME, PATH_DATASET_CNT
from utils_file_saver import save_df_to_disk
from utils_signal import SignalPreprocessor
from utils_feature_extraction import FeatureExtractor
from utils_env import FATIGUE_STR, FREQ, LOW_PASS_FILTER_RANGE_HZ, NOTCH_FILTER_HZ, NUM_USERS, SIGNAL_DURATION_SECONDS_DEFAULT, SIGNAL_OFFSET, get_brainwave_bands, feature_names, channels_good, training_columns_regex, driving_states

set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--user_num", metavar="N", type=int, help="Number of users that will be used (1 >= N <= 12)")
parser.add_argument("--signal-duration", metavar="N", type=int, help="Duration of the signal in seconds (1 >= N <= 300)")
parser.add_argument("--epoch-events-num", metavar="N", type=int, help="Each epoch will contain N events instead of using all 1000 events. The frequency is 1000HZ. (11 >= N <= 1000)")
parser.add_argument("--df-checkpoint", metavar="df", type=str, help="Load precaculated entropy dataframe (the one that isn't cleaned and normalized)")
parser.add_argument("--output-dir", metavar="dir", type=str, help="Directory where dataframe and npy files will be saved", default=PATH_DATAFRAME)
parser.add_argument("--use-brainbands", dest="use_brainbands", action="store_true", help="Decompose signal into alpha and beta bands")
parser.add_argument("--use-ica", dest="use_ica", action="store_true", help="Apply ICA for each subject")
parser.add_argument("--use-reref", dest="use_reref", action="store_true", help="Apply channel rereferencing")
parser.add_argument(
    "--channels_ignore", nargs="+", help="List of channels (electrodes) that will be ignored. Possible values: [HEOL, HEOR, FP1, FP2, VEOU, VEOL, F7, F3, FZ, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, CZ, C4, T4, TP7, CP3, CPZ, CP4, TP8, A1, T5, P3, PZ, P4, T6, A2, O1, OZ, O2, FT9, FT10, PO1, PO2]"
)

parser.set_defaults(user_num=NUM_USERS)
parser.set_defaults(signal_duration=SIGNAL_DURATION_SECONDS_DEFAULT)
parser.set_defaults(epoch_events_num=FREQ)
parser.set_defaults(brainbands=False)
parser.set_defaults(use_ica=False)
parser.set_defaults(use_reref=False)
parser.set_defaults(channels_ignore=[])
args = parser.parse_args()

user_num = args.user_num
signal_duration = args.signal_duration
epoch_events_num = args.epoch_events_num
output_dir = args.output_dir
use_brainbands = args.use_brainbands
use_ica = args.use_ica
use_reref = args.use_reref
channels_ignore = args.channels_ignore
channels = list(set(channels_good) - set(channels_ignore))

is_complete_train = not any(map(lambda arg_name: is_arg_default(arg_name, parser, args), ["user_num", "signal_duration", "epoch_events_num", "channels_ignore"]))
train_metadata = {"is_complete_train": is_complete_train, "brains": use_brainbands, "ica": use_ica, "reref": use_reref}
print("Training on {} dataset...".format("complete" if is_complete_train else "partial"))


def load_clean_cnt(filename: str, channels: List[str]):
    cnt = read_raw_cnt(filename, preload=True, verbose=False)
    print("Bad channels found by MNE:", cnt.info["bads"])
    cnt.pick_channels(channels)
    return cnt


def signal_crop(signal: BaseRaw, freq: float, signal_offset: float, signal_duration_wanted: float):
    signal_total_duration = floor(len(signal) / freq)
    start = signal_total_duration - signal_duration_wanted + signal_offset
    end = signal_total_duration + signal_offset
    return signal.crop(tmin=start, tmax=end)


def signal_filter_notch(signal: BaseRaw, filter_hz):
    return signal.copy().notch_filter(np.arange(filter_hz, (filter_hz * 5) + 1, filter_hz))


def low_high_pass_filter(signal: BaseRaw, l_freq, h_freq):
    """
    Filters the signal (cutoff lower and higher frequency) by using zero-phase filtering
    """
    return signal.copy().filter(l_freq=l_freq, h_freq=h_freq)


def epochs_to_dataframe(epochs: Epochs, drop_columns=["time", "condition"]):
    """
    Converts to dataframe and drops unnecessary columns
    """
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(drop_columns, axis=1)
    return df


def get_column_name(feature: str, channel: str, suffix: str = None):
    result = "_".join([channel, feature])
    result = result if suffix is None else "_".join([result, suffix])
    return result


def get_column_names(channels, feature_names, preprocess_procedure_names: dict):
    prod = product(channels, feature_names, preprocess_procedure_names)
    return list(map(lambda strs: "_".join(strs), prod))


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


feature_extractor = FeatureExtractor(picked_features=feature_names)

signal_preprocessor = SignalPreprocessor()

"""
Register signal preprocessing procedures with SignalPreprocessor
"standard" -> .notch and .filter with lower and higher pass-band edge defined in the research paper
"AL", "AH", "BL", "BH" -> .notch and .filter with lower and higher pass-band edge defined in brainwave_bands in env.py
"reref" -> .notch and .filter with lower and higher pass-band edge defined in the research paper and rereference within electodes
"""

base_preprocess_procedure = serialize_functions(
    lambda s: signal_crop(s, FREQ, SIGNAL_OFFSET, signal_duration),
    lambda s: signal_filter_notch(s, NOTCH_FILTER_HZ),
)

filter_frequencies = {"standard": LOW_PASS_FILTER_RANGE_HZ}
if use_brainbands:
    filter_frequencies.update(get_brainwave_bands())

for freq_name, freq_range in filter_frequencies.items():
    low_freq, high_freq = freq_range
    proc = serialize_functions(
        base_preprocess_procedure,
        lambda s: s.filter(low_freq, high_freq),
    )
    signal_preprocessor.add_preprocess_procedure(freq_name, proc, context={"freq_filter_range": freq_range})

if use_reref:
    low_freq, high_freq = LOW_PASS_FILTER_RANGE_HZ
    proc = serialize_functions(
        base_preprocess_procedure,
        lambda s: s.filter(low_freq, high_freq).set_eeg_reference(ref_channels="average", ch_type="eeg"),
    )
    signal_preprocessor.add_preprocess_procedure("reref", proc, context={"freq_filter_range": LOW_PASS_FILTER_RANGE_HZ})

training_cols = get_column_names(channels, feature_extractor.picked_features, signal_preprocessor.get_preprocess_procedure_names())
df_dict = {k: [] for k in ["is_fatigued", "user_id", "epoch_id", *training_cols]}

for user_id, driving_state in tqdm(list(product(range(0, user_num), driving_states))):
    is_fatigued = 1 if driving_state == FATIGUE_STR else 0
    file_signal = str(Path(PATH_DATASET_CNT, get_cnt_filename(user_id + 1, driving_state)))

    signal = load_clean_cnt(file_signal, channels)
    signal_preprocessor.fit(signal)
    for signal_processed, proc_name, proc_context in signal_preprocessor.get_preprocessed_signals():
        epochs = make_fixed_length_epochs(signal_processed)
        df = epochs_to_dataframe(epochs)
        freq_filter_range = proc_context["freq_filter_range"]
        feature_extractor.fit(signal_processed, FREQ)

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

            df_epoch = df.loc[df["epoch"] == epoch_id, channels].head(epoch_events_num)
            feature_dict = feature_extractor.get_features(df_epoch, context=dict(epoch_id=epoch_id, freq_filter_range=freq_filter_range))

            for channel_idx, channel in enumerate(channels):
                for feature_name, feature_array in feature_dict.items():
                    df_dict[get_column_name(feature_name, channel, proc_name)].append(feature_array[channel_idx])

            df_dict["user_id"].append(user_id)
            df_dict["is_fatigued"].append(is_fatigued)
            df_dict["epoch_id"].append(epoch_id)

"""Create dataframe from rows and columns"""
df = DataFrame.from_dict(df_dict)
df["is_fatigued"] = df["is_fatigued"].astype(int)
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
