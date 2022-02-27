"""
Creates a tabluar dataframe file which is used for training with the following form:

|     | is_fatigued | driver_id | epoch_id | CP3_PE_standard | CP3_PE_AL | ... | FT7_PE_standard |   FT7_PE_AL | ... |
| --- | ----------: | --------: | -------: | --------------: | --------: | --- | --------------: | ----------: | --- |
| 0   |           0 |         0 |        0 |        0.361971 |  0.361971 | ... |     1.84037e-23 | 1.84037e-23 | ... |
| 1   |           0 |         0 |        1 |        0.232837 |  0.232837 | ... |      1.4759e-23 |  1.4759e-23 | ... |
| 2   |           0 |         0 |        2 |        0.447734 |  0.447734 | ... |     1.27735e-23 | 1.27735e-23 | ... |
| 3   |           1 |         0 |        0 |         3.18712 |   3.18712 | ... |      1.4759e-23 |  1.4759e-23 | ... |
| 4   |           1 |         0 |        1 |         2.81654 |   2.81654 | ... |     1.27735e-23 | 1.27735e-23 | ... |

Number of rows:
drivers (NUM_DRIVERS=12) *
epochs (SIGNAL_DURATION_SECONDS_DEFAULT=300) *
driving_states (len(driving_states)=2)

12 * 300 * 2 = 7200 rows

Number of columns:
driver_id (1) +
is_fatigue_state (1) +
epoch_id (1) +
features (len(feature_names)=7) *
channels (len(channels_good)=30) *
preprocess procedures N (N <1, +>)

1 + 1 + 1 + 7 * 30 * N = 210 * N ~ in most cases it's 1050

The dataframe file is saved at ./data/dataframes by default
Dataframe files with prefix "complete-clean" should be used as the dataframe for training later on. 
"""

import argparse
import warnings
from itertools import product
from math import floor
from pathlib import Path
from typing import List, Union

import numpy as np
from mne import Epochs
from mne.epochs import make_fixed_length_epochs
from mne.io.base import BaseRaw
from mne.io.cnt import read_raw_cnt
from pandas import DataFrame, set_option
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocess_preprocess_df import df_replace_values
from utils_env import FATIGUE_STR, FREQ, LOW_PASS_FILTER_RANGE_HZ, NOTCH_FILTER_HZ, NUM_USERS, SIGNAL_DURATION_SECONDS_DEFAULT, SIGNAL_OFFSET, channels_good, driving_states, feature_names, get_brainwave_bands
from utils_feature_extraction import FeatureExtractor
from utils_file_saver import save_df
from utils_functions import get_cnt_filename, glimpse_df, is_arg_default, serialize_functions
from utils_paths import PATH_DATAFRAME, PATH_DATASET_CNT
from utils_signal import SignalPreprocessor

set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--driver_num", metavar="N", type=int, help="Number of drivers that will be used (1 >= N <= 12)")
parser.add_argument("--signal-duration", metavar="N", type=int, help="Duration of the signal in seconds (1 >= N <= 300)")
parser.add_argument("--epoch-events-num", metavar="N", type=int, help="Each epoch will contain N events instead of using all 1000 events. The frequency is 1000HZ. (11 >= N <= 1000)")
parser.add_argument("--df-checkpoint", metavar="df", type=str, help="Load precaculated entropy dataframe (the one that isn't cleaned and normalized)")
parser.add_argument("--output-dir", metavar="dir", type=str, help="Directory where dataframe and npy files will be saved", default=PATH_DATAFRAME)
parser.add_argument("--use-brainbands", dest="use_brainbands", action="store_true", help="Decompose signal into alpha and beta bands")
parser.add_argument("--use-reref", dest="use_reref", action="store_true", help="Apply channel rereferencing")
parser.add_argument(
    "--channels-ignore", nargs="+", help="List of channels (electrodes) that will be ignored. Possible values: [HEOL, HEOR, FP1, FP2, VEOU, VEOL, F7, F3, FZ, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, CZ, C4, T4, TP7, CP3, CPZ, CP4, TP8, A1, T5, P3, PZ, P4, T6, A2, O1, OZ, O2, FT9, FT10, PO1, PO2]"
)
parser.set_defaults(driver_num=NUM_USERS)
parser.set_defaults(signal_duration=SIGNAL_DURATION_SECONDS_DEFAULT)
parser.set_defaults(epoch_events_num=FREQ)
parser.set_defaults(brainbands=False)
parser.set_defaults(use_reref=False)
parser.set_defaults(channels_ignore=[])
args = parser.parse_args()

driver_num = args.driver_num
signal_duration = args.signal_duration
epoch_events_num = args.epoch_events_num
output_dir = args.output_dir
use_brainbands = args.use_brainbands
use_reref = args.use_reref
channels_ignore = args.channels_ignore
channels = list(set(channels_good) - set(channels_ignore))

is_complete_dataset = not any(map(lambda arg_name, parser=parser, args=args: is_arg_default(arg_name, parser, args), ["driver_num", "signal_duration", "epoch_events_num", "channels_ignore"]))
train_metadata = {"is_complete_dataset": is_complete_dataset, "brains": use_brainbands, "reref": use_reref}
print("Creating {} dataset...".format("complete" if is_complete_dataset else "partial"))


def load_clean_cnt(filename: str, channels: List[str]):
    cnt = read_raw_cnt(filename, preload=True, verbose=False)
    print("Bad channels found by MNE:", cnt.info["bads"])
    cnt.pick_channels(channels)
    return cnt


def signal_crop(signal: BaseRaw, freq: float, signal_offset: float, signal_duration_wanted: float):
    signal_total_duration = floor(len(signal) / freq)
    start = signal_total_duration - signal_duration_wanted + signal_offset
    end = signal_total_duration + signal_offset
    return signal.copy().crop(tmin=start, tmax=end)


def signal_filter_notch(signal: BaseRaw, filter_hz):
    return signal.copy().notch_filter(np.arange(filter_hz, (filter_hz * 5) + 1, filter_hz))


def low_high_pass_filter(signal: BaseRaw, l_freq, h_freq):
    """Filters the signal (cutoff lower and higher frequency) by using zero-phase filtering"""

    return signal.copy().filter(l_freq=l_freq, h_freq=h_freq)


def epochs_to_dataframe(epochs: Epochs, drop_columns=["time", "condition"]):
    """Converts to dataframe and drops unnecessary columns"""

    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(drop_columns, axis=1)
    return df


def get_column_name(feature: str, channel: str, suffix: Union[str, None] = None):
    result = "_".join([channel, feature])
    result = result if suffix is None else "_".join([result, suffix])
    return result


def get_column_names(channels, feature_names, preprocess_procedure_names: List[str]):
    prod = product(channels, feature_names, preprocess_procedure_names)
    return list(map(lambda strs: "_".join(strs), prod))


feature_extractor = FeatureExtractor(selected_feature_names=feature_names)
signal_preprocessor = SignalPreprocessor()

"""
4 signal preprocessing procedures with SignalPreprocessor
"standard" -> .notch and .filter with lower and higher pass-band edge defined in the research paper

"AL", "AH", "BL", "BH" -> .notch and .filter with lower and higher pass-band edge defined in brainwave_bands in env.py

"reref" -> .notch and .filter with lower and higher pass-band edge defined in the research paper and rereference within electodes
"""


filter_frequencies = {"standard": LOW_PASS_FILTER_RANGE_HZ}
if use_brainbands:
    filter_frequencies.update(get_brainwave_bands())


""" Registers preprocessing procedures that will filter frequencies defined in filter_frequencies"""
for freq_name, freq_range in filter_frequencies.items():
    low_freq, high_freq = freq_range

    procedure = serialize_functions(
        lambda s, notch_filter_hz=NOTCH_FILTER_HZ: signal_filter_notch(s, notch_filter_hz),
        lambda s, low_freq=low_freq, high_freq=high_freq: s.copy().filter(low_freq, high_freq),
        lambda s, freq=FREQ, signal_offset=SIGNAL_OFFSET, signal_duration=signal_duration: signal_crop(s, freq, signal_offset, signal_duration),
    )

    signal_preprocessor.register_preprocess_procedure(freq_name, procedure=procedure, context={"freq_filter_range": freq_range})

""" Registers preprocessing procedure that uses channel rereferencing"""
if use_reref:
    low_freq, high_freq = LOW_PASS_FILTER_RANGE_HZ
    proc = serialize_functions(
        lambda s, notch_filter_hz=NOTCH_FILTER_HZ: signal_filter_notch(s, notch_filter_hz),
        lambda s, low_freq=low_freq, high_freq=high_freq: s.filter(low_freq, high_freq).set_eeg_reference(ref_channels="average", ch_type="eeg"),
        lambda s, freq=FREQ, signal_offset=SIGNAL_OFFSET, signal_duration=signal_duration: signal_crop(s, freq, signal_offset, signal_duration),
    )
    signal_preprocessor.register_preprocess_procedure("reref", proc, context={"freq_filter_range": LOW_PASS_FILTER_RANGE_HZ})

training_cols = get_column_names(channels, feature_extractor.get_feature_names(), signal_preprocessor.get_preprocess_procedure_names())
df_dict = {k: [] for k in ["is_fatigued", "driver_id", "epoch_id", *training_cols]}

for driver_id, driving_state in tqdm(list(product(range(0, driver_num), driving_states))):

    is_fatigued = 1 if driving_state == FATIGUE_STR else 0
    signal_filepath = str(Path(PATH_DATASET_CNT, get_cnt_filename(driver_id + 1, driving_state)))

    signal = load_clean_cnt(signal_filepath, channels)
    signal_preprocessor.fit(signal)
    for proc_index, (signal_processed, proc_name, proc_context) in tqdm(enumerate(signal_preprocessor.get_preprocessed_signals())):
        epochs = make_fixed_length_epochs(signal_processed, verbose=False)
        df = epochs_to_dataframe(epochs)

        freq_filter_range = proc_context["freq_filter_range"]
        feature_extractor.fit(signal_processed, FREQ)
        for epoch_id in tqdm(range(0, signal_duration)):
            """
            Filter the dataframe rows by selecting the rows with epoch_id.

            get_features caculates all features for a given epoch
                * each feature is caculated for all channels
                * for 30 channels the shape of a feature will be (30,) (if feature is e.g. float)
                * feature_dict dictionary contains all features

            Features are flattened to a combination (channel, feature, preprocess procedure) which defines a single column
                * e.g. F4_std_standard
                * total number of columns is (len(features) * len(channels) * len(preprocess_procedures))

            For loop appends the result to each column. Effectively, it's itterating through the feature array (30,) and appends each float to appropriate column
                * once the for loop is completed, an appropriate result will be appended to each column
            """

            df_epoch = df.loc[df["epoch"] == epoch_id, channels].head(epoch_events_num)
            feature_dict = feature_extractor.get_features(df_epoch, epoch_id=epoch_id, freq_filter_range=freq_filter_range)

            for channel_idx, channel in enumerate(channels):
                for feature_name, feature_array in feature_dict.items():
                    df_dict[get_column_name(feature_name, channel, proc_name)].append(feature_array[channel_idx])
            if proc_index == 0:
                df_dict["epoch_id"].append(epoch_id)
                df_dict["driver_id"].append(driver_id)
                df_dict["is_fatigued"].append(is_fatigued)
    """
    Checkpoint - save the dataset after each driver anddriving state combination
    """
    if is_complete_dataset:
        tmp_df = DataFrame.from_dict(df_dict)
        tmp_df["is_fatigued"] = tmp_df["is_fatigued"].astype(int)
        tmp_df["driver_id"] = tmp_df["driver_id"].astype(int)
        tmp_df["epoch_id"] = tmp_df["epoch_id"].astype(int)
        tmp_df.to_pickle(str(Path(output_dir, ".raw_df.pkl")))

"""Create dataframe from rows and columns"""
df = DataFrame.from_dict(df_dict)
df["is_fatigued"] = df["is_fatigued"].astype(int)
df["driver_id"] = df["driver_id"].astype(int)
df["epoch_id"] = df["epoch_id"].astype(int)
df.to_pickle(str(Path(output_dir, ".raw_df.pkl")))

"""Save to files"""
save_df(df, is_complete_dataset, output_dir, "raw", train_metadata)
glimpse_df(df)
df = df_replace_values(df)
save_df(df, is_complete_dataset, output_dir, "clean", train_metadata)
