"""
Find out which entropy combinations performs the best 


Load the dataset with the --df argument
Create all possible combinations of entropies:
[('PE'), ('AE'), ('SE'), ('FE'), ('PE', 'AE'), ('PE', 'SE'), ('PE', 'FE'), ('AE', 'SE'), ('AE', 'FE'), ('SE', 'FE'), ('PE', 'AE', 'SE'), ('PE', 'AE', 'FE'), ('PE', 'SE', 'FE'), ('AE', 'SE', 'FE'), ('PE', 'AE', 'SE', 'FE')]
Train GridSearch SVM model on each entropy combination
Create a report file
"""
import argparse
import pickle
import warnings
from datetime import datetime
from itertools import chain, combinations, product
from pathlib import Path

from IPython.core.display import display
from joblib import dump, load
from pandas import DataFrame, read_pickle
from pandas._config.config import set_option
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from model import model_svc
from utils_env import entropy_names
from utils_file_saver import save_model
from utils_functions import get_dictionary_leaves, get_timestamp, glimpse_df, powerset, stdout_to_file
from utils_paths import PATH_REPORT

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("-r", "--output-report", metavar="dir", required=False, type=str, help="Directory where report file will be created.", default=PATH_REPORT)
args = parser.parse_args()
stdout_to_file(Path(args.output_report, "-".join(["best-entropies", get_timestamp()]) + ".txt"))


df: DataFrame = read_pickle(args.df)
X = df.loc[:, ~df.columns.isin(["is_fatigued"])]
y = df.loc[:, "is_fatigued"]

X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

entropy_excluded_powerset = list(powerset(entropy_names))[:-1]  # exclude last element (PE, AE, FE, SE)
models = [model_svc]
scorings = ["accuracy"]
results = []

for i, pair in enumerate(tqdm(list(product(scorings, models, entropy_excluded_powerset)))):
    scoring, model, entropies_exclude = pair
    (X_train, X_test) = (X_train_org.copy(), X_test_org.copy())

    for entropy in entropies_exclude:
        X_train = X_train.loc[:, ~X_train.columns.str.contains(entropy)]
        X_test = X_test.loc[:, ~X_test.columns.str.contains(entropy)]

    model.scoring = scoring
    model.fit(X_train, y_train)

    y_true_train, y_pred_train = y_train, model.predict(X_train)
    y_true_test, y_pred_test = y_test, model.predict(X_test)

    classification_report_string = classification_report(y_true_test, y_pred_test, digits=6, output_dict=True)

    results.append([i, list(set(entropy_names) - set(entropies_exclude)), model.best_score_, get_dictionary_leaves(classification_report_string)])

for result in sorted(results, key=lambda x: x[2], reverse=True):
    print(result[0], result[1], result[2])
