from math import floor
from mne.io import read_raw_cnt
from pathlib import Path

from utils_paths import *
from utils_env import *

filename = str(Path(PATH_DATASET_CNT, "2_fatigue.cnt"))
eeg = read_raw_cnt(filename, verbose=False)
eeg_filtered = eeg.load_data(verbose=False).filter(l_freq=0.15, h_freq=40).notch_filter(50)
signal_seconds_floored = floor(len(eeg_filtered) / FREQ)
tmin = signal_seconds_floored - signal_duration_seconds - signal_offset_s
tmax = signal_seconds_floored - signal_offset_s
eeg_filtered = eeg_filtered.crop(tmin=tmin, tmax=tmax)
eeg.plot()
