from mne.epochs import Epochs
from preprocess_unzip_data import unzip_data
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import sys
from datetime import datetime

from utils_paths import *
from utils_env import *
from utils_functions import *
from utils_entropy import *

set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--users",
    metavar="N",
    type=int,
    help="Number of users that will be used (<= 12)",
)

parser.add_argument(
    "--electrodes",
    metavar="N",
    type=int,
    nargs=2,
    help="Range of electrodes that will be used (N <= 30)",
)

parser.add_argument(
    "--sig", metavar="N", type=int, help="Duration of signal (<= 300)"
)
parser.add_argument(
    "--epoch_elems", metavar="N", type=int, help="Duration of signal (<= 1000)"
)

parser.add_argument("--unzip", metavar="N", type=bool, help="True/False")
args = parser.parse_args()

is_complete_itteration = len(sys.argv) == 1


USER_COUNT = args.users if (args.users) else USER_COUNT
SIGNAL_REQUESTED_SECONDS = args.sig if (args.sig) else SIGNAL_REQUESTED_SECONDS
if args.electrodes:
    subset = elect_cols[args.electrodes[0] : args.electrodes[1]]
    elect_cols_ignore = [
        electrode for electrode in elect_cols if electrode not in subset
    ]
    elect_cols = subset

epoch_elems = args.epoch_elems if args.epoch_elems else FREQ

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
    tmin = (
        signal_seconds_floored
        - SIGNAL_REQUESTED_SECONDS
        - SAFETY_CUTOFF_SECONDS
    )
    tmax = signal_seconds_floored - SAFETY_CUTOFF_SECONDS
    eeg_filtered = eeg_filtered.crop(tmin=tmin, tmax=tmax)
    return make_fixed_length_epochs(
        eeg, duration=EPOCH_SECONDS, preload=False, verbose=False
    )


def get_df_from_epochs(epochs: Epochs, state: str):
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1, mag=1, grad=1))
    df = df.drop(["time", "condition", *elect_cols_ignore], axis=1)
    return df


# {(0,normal), (0,fatigue), (1,normal)...(12,fatigue)}
user_state_pairs = [
    (i_user, state)
    for i_user in range(0, USER_COUNT)
    for state in [NORMAL_STR, FATIGUE_STR]
]

arr_users = []
entropy_order = ["PE", "AE", "SE", "FE"]

for pair in user_state_pairs:
    print(pair)
    i_user, state = pair
    filename = str(Path(PATH_DATA_CNT, get_cnt_filename(i_user + 1, state)))
    epochs = get_epochs_from_signal(filename)
    df = get_df_from_epochs(epochs, state)
    label = 1 if state == FATIGUE_STR else 0

    for i_poch in range(0, SIGNAL_REQUESTED_SECONDS):
        print(i_poch)
        # filter rows for current epoch
        # calculate all entropies for all electrodes
        # store them in a dictionary so they can be ordered correctly
        # return list with label + ordered entropies

        df_dict = {}
        df_epoch = df.loc[df["epoch"] == i_poch].head(epoch_elems)
        df_electrodes = df_epoch[elect_cols]

        df_spectral_entropy = df_electrodes.apply(
            func=lambda x: pd_spectral_entropy(
                x, freq=FREQ, standardize_input=True
            ),
            axis=0,
        )  # type: ignore
        df_approximate_entropy = df_electrodes.apply(
            func=lambda x: pd_approximate_entropy(x, standardize_input=True),
            axis=0,
        )  # type: ignore
        df_sample_entropy = df_electrodes.apply(
            func=lambda x: pd_sample_entropy(x, standardize_input=True), axis=0
        )  # type: ignore
        df_fuzzy_entropy = df_electrodes.apply(
            func=lambda x: pd_fuzzy_entropy(x, standardize_input=True), axis=0
        )  # type: ignore

        df_dict = {
            "PE": df_spectral_entropy,
            "AE": df_approximate_entropy,
            "SE": df_sample_entropy,
            "FE": df_fuzzy_entropy,
        }

        # [0, PE_FP1, PE_FP2, ... , PE_C3, AE_FP1, AE_FP2, ..., FE_C3]
        arr_users.append(
            [
                label,
                *df_dict[entropy_order[0]],
                *df_dict[entropy_order[1]],
                *df_dict[entropy_order[2]],
                *df_dict[entropy_order[3]],
            ]
        )

entropy_electrode_combinations = [
    "{}_{}".format(entropy, electrode)
    for entropy in entropy_order
    for electrode in elect_cols
]

columns = ["label"] + entropy_electrode_combinations
df = DataFrame(arr_users, columns=columns)
save_df_to_disk(df, is_complete_itteration, PATH_DATA_PCKL)


def clean_df(df: DataFrame):
    # set to 0 if treshold is met
    # NaN entropies can be set to zero
    # standard scaler scales for each column independently
    threshold = 1e-6
    df[df < threshold] = 0
    df[df <= 0] = 0
    df = df.fillna(0)
    df[entropy_electrode_combinations] = min_max_scaler(
        df[entropy_electrode_combinations]
    )
    return df


df = clean_df(df)
glimpse_df(df)

X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]

scores = ["precision", "recall"]


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, scoring="%s_macro" % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
