import antropy as an
import EntropyHub as eh
import numpy as np
from pandas import Series
from env import *
from utils import *


def fuzzy_entropy(x): return eh.FuzzEn(
    x, m=2, r=(np.std(x, ddof=0) * 0.2, 1))[0][-1]


def sample_entropy(x): return an.sample_entropy(x)


# don't normalize because you have to normalze across all users and not based on 1 user and 1 sample
def spectral_entropy(x): return an.spectral_entropy(
    x, sf=FREQ, normalize=False)


def approximate_entropy(x): return an.app_entropy(x, order=2)


def pd_fuzzy_entropy(x: Series, standardize_input=False) -> float:
    # standardization doesnt affect result!
    x_np = x.to_numpy()
    if standardize_input:
        x_np = standard_scaler_1d(x_np)
    return fuzzy_entropy(x_np)


def pd_sample_entropy(x: Series, standardize_input=False) -> float:
    # standardization doesnt affect result!
    x_np = x.to_numpy()
    if standardize_input:
        x_np = standard_scaler_1d(x_np)
    return sample_entropy(x_np)


def pd_spectral_entropy(x: Series, standardize_input=False) -> float:
    # standardization doesnt affect result!
    x_np = x.to_numpy()
    if standardize_input:
        x_np = standard_scaler_1d(x_np)
    return spectral_entropy(x_np)


def pd_approximate_entropy(x: Series, standardize_input=False) -> float:
    x_np = x.to_numpy()
    if standardize_input:
        x_np = min_max_scaler_1d(x_np)
    return approximate_entropy(x_np)
