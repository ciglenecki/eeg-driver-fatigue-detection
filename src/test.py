import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from zipfile import ZipFile
from pathlib import Path
from os import getcwd, listdir, path
import os
import scipy.io
import warnings
from typing import Dict, Generic
from sklearn import preprocessing
from typing import TypeVar

warnings.filterwarnings("ignore")

UNZIP_DATA = False

PATH_CWD = Path(getcwd())

PATH_DATA = Path(PATH_CWD, "data")
PATH_DATA_MAT = Path(PATH_DATA, "mat")
PATH_DATA_CNT = Path(PATH_DATA, "cnt")

PATH_ZIP_CNT = Path(PATH_DATA, "5202739.zip")
PATH_ZIP_MAT = Path(PATH_DATA, "5202751.zip")


raw = mne.io.read_raw_cnt(str(Path(PATH_DATA_CNT, "1_fatigue.cnt")))
print(raw.info)
print(raw.get_data())
# ica = mne.preprocessing.ICA(n_components=2, random_state=97, max_iter=800)

# ica.fit(raw)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(raw, picks=ica.exclude)
raw.plot()
raw.plot_psd(
    fmin=0,
    tmin=0,
    tmax=1,
    exclude=[],
)
# raw.plot(duration=5, )
