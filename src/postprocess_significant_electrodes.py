import argparse
from itertools import chain, combinations
from os import getcwd
from pathlib import Path
import warnings
from IPython.core.display import display
from pandas import read_pickle
from pandas._config.config import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import pickle
from joblib import dump, load
from utils_file_saver import load_model, save_model
from utils_functions import glimpse_df, powerset, stdout_to_file
from utils_paths import PATH_DATA_MODEL
from itertools import product
from model import model_rfc, model_mlp, model_svc, model_knn
from utils_env import channels_good
from itertools import combinations
from tqdm import tqdm
import sys

"""
Get the trained model.
Get the dataset
Since we don't know which data was used in training we will refit the model with same optimal parameters.
Caculate weight for every channel and sort by weight.
"""

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("--svm", metavar="file", required=True, type=str, help="SVM model used for caclulating the accuracy")
parser.add_argument("--report", metavar="file", required=True, type=str, help="File where report will be saved")
args = parser.parse_args()
stdout_to_file(Path(args.report))

model: SVC = load_model(args.svm).best_estimator_
model.scoring = "accuracy"

df = read_pickle(args.df)
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]
X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

channels_pairs = list(combinations(channels_good, 2))
channel_acc = {}

"""
Calculate single accuracy for each channel (Acc_i).
"""
for ch in channels_good:
    X_train = X_train_org.loc[:, X_train_org.columns.str.contains(ch)]
    X_test = X_test_org.loc[:, X_test_org.columns.str.contains(ch)]
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    channel_acc[ch] = accuracy_score(y_test, y_test_pred)

"""
Calculate weight for each channel (V_i).
"""
channel_weights = []
for i, channels_i in enumerate(tqdm(channels_good)):
    acc_dual = 0
    for channels_j in channels_good:
        if channels_j == channels_i:
            break
        X_train = X_train_org.loc[:, X_train_org.columns.str.contains("|".join([channels_i, channels_j]))]
        X_test = X_test_org.loc[:, X_test_org.columns.str.contains("|".join([channels_i, channels_j]))]
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)

        acc_ij = accuracy_score(y_test, y_test_pred)
        acc_dual += acc_ij + channel_acc[channels_i] - channel_acc[channels_j]

    weight = (channel_acc[channels_i] + acc_dual) / len(channels_good)
    channel_weights.append([channels_i, weight])

channel_weights = sorted(channel_weights, key=lambda x: x[1], reverse=True)
print(channel_weights)
sys.stdout.close()
