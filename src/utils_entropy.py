import antropy as an
import EntropyHub as eh
import numpy as np
from pandas import Series
from utils_functions import *


def fuzzy_entropy(x):
    return eh.FuzzEn(x, m=2, r=(np.std(x, ddof=0) * 0.2, 1))[0][-1]


def sample_entropy(x):
    return an.sample_entropy(x)


# don't normalize because you have to normalze across all users and not based on 1 user and 1 sample
def spectral_entropy(x, freq: float):
    return an.spectral_entropy(x, sf=freq, normalize=False)


def approximate_entropy(x):
    return an.app_entropy(x, order=2)


def pd_fuzzy_entropy(x: Series) -> float:
    return fuzzy_entropy(x.to_numpy())


def pd_sample_entropy(x: Series) -> float:
    return sample_entropy(x_np=x.to_numpy())


def pd_spectral_entropy(x: Series, freq: float) -> float:
    return spectral_entropy(x.to_numpy(), freq)


def pd_approximate_entropy(x: Series) -> float:
    return approximate_entropy(x.to_numpy())
