"""
Functions which caculate 4 entropies
"""
import antropy as an
import EntropyHub as eh
import numpy as np
from pandas import Series
import pandas
from scipy.signal.filter_design import normalize
from scipy.signal.spectral import periodogram
from utils_functions import *
from scipy import signal


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


def pd_fuzzy_entropy(x: Series) -> float:
    return fuzzy_entropy(x.to_numpy())


def pd_sample_entropy(x: Series) -> float:
    return sample_entropy(x.to_numpy())


def pd_spectral_entropy(x: Series, freq: float) -> float:
    return spectral_entropy(x.to_numpy(), freq)


def pd_approximate_entropy(x: Series) -> float:
    return approximate_entropy(x.to_numpy())


def psd_welch(x: Series):
    _, psd = signal.welch(x)
    return psd
