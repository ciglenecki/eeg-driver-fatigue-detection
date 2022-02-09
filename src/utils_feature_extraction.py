"""
Functions which caculate 4 entropies
"""
import antropy as an
import EntropyHub as eh
import numpy as np
from pandas import Series, DataFrame
import pandas
from scipy.signal.filter_design import normalize
from scipy.signal.spectral import periodogram
from utils_functions import *
from scipy import signal
from typing import List, Optional, TypedDict, Union, Tuple
from typing import Callable
from mne import make_fixed_length_epochs, time_frequency
from math import ceil, floor
from mne.io.base import BaseRaw


feature_long_to_short_mapping = {
    "mean": "mean",
    "standard_deviation": "std",
    "power_spectral_density": "psd",
    "spectral_entropy": "PE",
    "approximate_entropy": "AE",
    "sample_entropy": "SE",
    "fuzzy_entropy": "FE",
}


def fuzzy_entropy(x):
    return eh.FuzzEn(x, m=2, r=(np.std(x, ddof=0) * 0.2, 1))[0][-1]


def sample_entropy(x):
    return an.sample_entropy(x)


# don't normalize because you have to normalze across all users and not based on 1 user and 1 sample
def spectral_entropy(x, freq: float):
    axis = -1
    sf = freq
    normalize = False

    x = np.asarray(x)
    _, psd = periodogram(x, sf, axis=axis)
    psd_norm = psd[1:] / psd[1:].sum(axis=axis, keepdims=True)
    se = -(psd_norm * np.log2(psd_norm)).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se


def approximate_entropy(x):
    return an.app_entropy(x, order=2)


def psd_welch(x: Series):
    _, psd = signal.welch(x)
    return psd


class FeatureContext(TypedDict):
    freq_signal: Optional[float]
    freq_filter_range: Optional[Tuple[float, float]]
    epoch_id: Optional[int]


class FeatureExtractorFeatureContextError(Exception):
    pass


class FeatureExtractorPreprocessProcedureNotRegistered(Exception):
    pass


class FeatureExtractor:
    def __init__(self, picked_features: List[str]):
        self.signal = None
        self.freq = None
        self.psds = None
        self.picked_features = picked_features

    def _validate_feature_context(self, key, context: FeatureContext):
        if key not in context:
            raise FeatureExtractorFeatureContextError("Missing key '{}' in the context.".format(key))

    def fit(self, signal, freq):
        self.signal = signal
        self.freq = freq
        if "psd" in self.picked_features:
            epochs = make_fixed_length_epochs(signal)
            self.psds, _ = time_frequency.psd_welch(epochs, n_fft=freq, n_per_seg=freq, n_overlap=0, verbose=False)

    def get_features(self, df: DataFrame, context: FeatureContext = {}) -> Dict:
        """
        Create dictionary of features for dataframe
        """

        features = {}
        for key in self.picked_features:
            if key == "mean":
                features[key] = df.apply(func=lambda x: np.mean(x), axis=0)
                continue
            if key == "std":
                features[key] = df.apply(func=lambda x: np.std(x), axis=0)
                continue
            if key == "psd":
                self._validate_feature_context("epoch_id", context)
                epoch_id = context["epoch_id"]
                series = self.psds[epoch_id, :]
                if "freq_filter_range" in context:
                    low_freq, high_freq = context["freq_filter_range"]
                    series = series[:, floor(low_freq) : ceil(high_freq)]
                features[key] = series.mean(axis=1)
                continue
            if key == "PE":
                features[key] = df.apply(func=lambda x: spectral_entropy(x.to_numpy(), self.freq), axis=0)
                continue
            if key == "AE":
                features[key] = df.apply(func=lambda x: approximate_entropy(x.to_numpy()), axis=0)
                continue
            if key == "SE":
                features[key] = df.apply(func=lambda x: sample_entropy(x.to_numpy()), axis=0)
                continue
            if key == "FE":
                features[key] = df.apply(func=lambda x: fuzzy_entropy(x.to_numpy()), axis=0)
                continue
        return features
