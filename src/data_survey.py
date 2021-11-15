
from unzip_data import unzip_data
from mne import make_fixed_length_epochs
import argparse
from time import time
import antropy as an
from math import floor
import EntropyHub as eh
from IPython.display import display, HTML
from typing import TypeVar
from sklearn import preprocessing
from mne.io import read_raw_cnt
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, set_option
from pathlib import Path
import warnings
from typing import Dict

from paths import *
from env import *
from utils import *

set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
T = TypeVar('T')  # Any

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--users', metavar='N', type=int,
                    help='Number of users that will be used (<= 12)')

parser.add_argument('--electrodes', metavar='N', type=int, nargs=2,
                    help='Range of electrodes that will be used (N <= 30)')

parser.add_argument('--sig', metavar='N', type=int,
                    help='Duration of signal (<= 300)')
parser.add_argument('--unzip', metavar='N', type=bool,
                    help='True/False')
args = parser.parse_args()


USER_COUNT = args.users if (args.users) else USER_COUNT
SIGNAL_REQUESTED_SECONDS = args.sig if (
    args.sig) else SIGNAL_REQUESTED_SECONDS
ELECTRODE_NAMES = ELECTRODE_NAMES[args.electrodes[0]:args.electrodes[1]] if (
    args.electrodes) else ELECTRODE_NAMES
ELECTRODE_NAMES = ELECTRODE_NAMES[args.electrodes[0]:args.electrodes[1]] if (
    args.electrodes) else ELECTRODE_NAMES

if args.unzip:
    unzip_data()


# In addition each BCIT dataset includes 4 additional EOG channels placed vertically above the right eye (veou), vertically below the right eye (veol), horizontally on the outside of the right eye (heor), and horizontally on the outside of the left eye (heol)


def fuzzy_entropy(x): return eh.FuzzEn(
    x, m=2, r=(np.std(x, ddof=0) * 0.2, 1))[0][-1]


def sample_entropy(x): return an.sample_entropy(x)


# don't normalize because you have to normalze across all users and not based on 1 user and 1 sample
def spectral_entropy(x): return an.spectral_entropy(
    x, sf=FREQ, normalize=False)


def approximate_entropy(x): return an.app_entropy(x, order=2)


def pd_fuzzy_entropy(x: Series, standardize_input=False):
    # standardization doesnt affect result!
    x = x.to_numpy()
    if standardize_input:
        x = standard_scaler_1d(x)
    return fuzzy_entropy(x)


def pd_sample_entropy(x: Series, standardize_input=False):
    # standardization doesnt affect result!
    x = x.to_numpy()
    if standardize_input:
        x = standard_scaler_1d(x)
    return sample_entropy(x)


def pd_spectral_entropy(x: Series, standardize_input=False):
    # standardization doesnt affect result!
    x = x.to_numpy()
    if standardize_input:
        x = standard_scaler_1d(x)
    return spectral_entropy(x)


def pd_approximate_entropy(x: Series, standardize_input=False):
    x = x.to_numpy()
    if standardize_input:
        x = min_max_scaler_1d(x)
    return approximate_entropy(x)

# SHOWCASE SIGNAL

# filename = str(Path(PATH_DATA_CNT, "2_fatigue.cnt"))
# eeg = read_raw_cnt(filename,verbose=False)
# eeg_filtered = eeg.load_data(verbose=False).filter(l_freq=0.15, h_freq=40).notch_filter(50)
# signal_seconds_floored =  floor(len(eeg_filtered) / FREQ)
# tmin = signal_seconds_floored - SIGNAL_REQUESTED_SECONDS - SAFETY_CUTOFF_SECONDS
# tmax = signal_seconds_floored - SAFETY_CUTOFF_SECONDS
# eeg_filtered = eeg_filtered.crop(tmin=tmin, tmax=tmax)
# eeg.plot()


def get_cnt_filename(i_user: int, state: str):
    return "{i_user}_{state}.cnt".format(i_user=i_user, state=state)


# {(0,normal), (0,fatigue), (1,normal)...(12,fatigue)}
user_state_pairs = [(i_user, state) for i_user in range(
    0, USER_COUNT) for state in [NORMAL_STR, FATIGUE_STR]]

arr_total = []
for pair in user_state_pairs:
    i_user, state = pair
    print(i_user)
    filename = str(Path(PATH_DATA_CNT, get_cnt_filename(i_user + 1, state)))

    # LOAD, FILTER, CROP AND EPOCH SIGNAL
    eeg = read_raw_cnt(
        filename, eog=['HEOL', "HEOR", "VEOU", "VEOL"], verbose=False)
    eeg_filtered = eeg.load_data(verbose=False).notch_filter(
        50).filter(l_freq=0.15, h_freq=40)
    signal_seconds_floored = floor(len(eeg_filtered) / FREQ)
    tmin = signal_seconds_floored - SIGNAL_REQUESTED_SECONDS - SAFETY_CUTOFF_SECONDS
    tmax = signal_seconds_floored - SAFETY_CUTOFF_SECONDS
    eeg_filtered = eeg_filtered.crop(tmin=tmin, tmax=tmax)
    epochs = make_fixed_length_epochs(
        eeg, duration=EPOCH_SECONDS, preload=False, verbose=False)

    # CREATE DF
    df: DataFrame = epochs.to_data_frame(
        scalings=dict(eeg=1, mag=1, grad=1))
    df['condition'] = df['condition'].astype(int)
    df.drop('time', axis=1, inplace=True)

    arr_one_user_samples = []
    for i_poch in range(0, SIGNAL_REQUESTED_SECONDS):
        # take epooch rows, divide electordes and info
        df_epoch = df.loc[df["epoch"] == i_poch]
        df_info: DataFrame = df_epoch.iloc[0,
                                           ~df_epoch.columns.isin(ELECTRODE_NAMES)]
        df_electrodes: DataFrame = df_epoch[ELECTRODE_NAMES]
        df_pe_en = df_electrodes.apply(
            func=lambda x: pd_spectral_entropy(x, standardize_input=True), axis=0)

        df_ae_en = df_electrodes.apply(
            func=lambda x: pd_approximate_entropy(x, standardize_input=True), axis=0)

        df_se_en = df_electrodes.apply(
            func=lambda x: pd_sample_entropy(x, standardize_input=True), axis=0)

        df_fe_en = df_electrodes.apply(
            func=lambda x: pd_fuzzy_entropy(x, standardize_input=True), axis=0)
        arr_one_user_samples.append(
            [*df_info, *df_pe_en, *df_ae_en, *df_se_en, *df_fe_en])
    arr_total.append(arr_one_user_samples)

new_df = DataFrame()
