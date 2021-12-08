import argparse
from itertools import chain, combinations
from pathlib import Path
import warnings
from IPython.core.display import display
from pandas import read_pickle, DataFrame
from pandas._config.config import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
from joblib import dump, load
from utils_env import entropy_names
from utils_file_saver import save_model
from utils_functions import get_dictionary_leaves, glimpse_df, powerset, stdout_to_file
from utils_paths import PATH_DATA_MODEL, PATH_REPORT
from itertools import product
from model import model_svc
from tqdm import tqdm
from datetime import datetime

set_option("display.max_columns", None)

parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
args = parser.parse_args()

timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
report_filename = Path(PATH_REPORT, "-".join(["best-entropies", timestamp]) + ".txt")
stdout_to_file(report_filename)


df: DataFrame = read_pickle(args.df)
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]

X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

entropy_excluded_powerset = list(powerset(entropy_names))[:-1]  # Exclude last element where all entropies are mentioned
models = [model_svc]
scorings = ["accuracy"]
results = []

for i, pair in enumerate(tqdm(list(product(scorings, models, entropy_excluded_powerset)))):
    scoring, model, entropies_exclude = pair
    (X_train, X_test) = (X_train_org.copy(), X_test_org.copy())

    for entropy in entropies_exclude:
        X_train = X_train.loc[:, ~X_train.columns.str.startswith(entropy)]
        X_test = X_test.loc[:, ~X_test.columns.str.startswith(entropy)]

    model.scoring = scoring
    model.fit(X_train, y_train)

    y_true_train, y_pred_train = y_train, model.predict(X_train)
    y_true_test, y_pred_test = y_test, model.predict(X_test)

    classification_report_string = classification_report(y_true_test, y_pred_test, digits=6, output_dict=True)

    results.append([i, list(set(entropy_names) - set(entropies_exclude)), model.best_score_, get_dictionary_leaves(classification_report_string)])

for result in sorted(results, key=lambda x: x[2], reverse=True):
    print(result, "\n")
