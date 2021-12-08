import argparse
from itertools import chain, combinations
from pathlib import Path
import warnings
from IPython.core.display import display
from pandas import read_pickle
from pandas._config.config import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
from joblib import dump, load
from tqdm import tqdm
from utils_env import PAPER_BF_HIDDEN, PAPER_C, PAPER_G, PAPER_RFC_INPUT_VARIABLES, PAPER_RFC_TREES
from utils_file_saver import save_model
from sklearn.neural_network import MLPClassifier
from utils_functions import glimpse_df, min_max_dataframe, powerset
from utils_paths import PATH_DATA_MODEL, PATH_DATASET_MAT
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from os import getcwd
import scipy.io
import pandas as pd
from utils_functions import min_max_scaler
from time import sleep

parser = argparse.ArgumentParser()
# parser.add_argument("--model", metavar="model", required=True, type=str, help="Model")
args = parser.parse_args()

# print(Path(getcwd(), args.model))

for i in tqdm(range(1, 500)):
    sleep(0.01)
    print(i)
