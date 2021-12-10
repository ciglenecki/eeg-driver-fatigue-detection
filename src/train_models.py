import argparse
from itertools import chain, combinations
from pathlib import Path
import warnings
from IPython.core.display import display
from pandas import read_pickle
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
from joblib import dump, load
from utils_file_saver import save_model
from utils_functions import get_timestamp, glimpse_df, powerset, min_max_scaler, stdout_to_file
from utils_paths import PATH_MODEL, PATH_REPORT
from itertools import product
from model import model_rfc, model_mlp, model_svc, model_knn
from tqdm import tqdm
from utils_env import entropy_channel_combinations

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
args = parser.parse_args()

stdout_to_file(Path(PATH_REPORT, "-".join(["train-models", get_timestamp()]) + ".txt"))


df = read_pickle(args.df)
glimpse_df(df)

df = df.loc[:, df.columns.isin(["label", *entropy_channel_combinations])]
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

scorings = ["accuracy"]  # scorings = ["accuracy", "f1"]
models = [model_rfc, model_mlp, model_knn, model_svc]
for pair in product(scorings, models):
    scoring, model = pair

    model_name = type(model.estimator).__name__
    model.scoring = scoring
    model.fit(X_train, y_train)
    save_model(model=model, model_name=model_name, score=model.best_score_, directory=PATH_MODEL, metadata=model.best_params_)

    print("=== Best model {} with accuracy {} and parameters {}\n\n".format(model_name, model.best_score_, model.best_params_))
    print("Grid scores on test set:\n")
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params in sorted(zip(means, stds, model.cv_results_["params"]), key=lambda x: x[0]):
        print("%0.6f (+/-%0.6f) for %r" % (mean, std * 2, params))

    y_true_train, y_pred_train = y_train, model.predict(X_train)
    y_true_test, y_pred_test = y_test, model.predict(X_test)

    print("\nReport on train set:")
    classification_report_string = classification_report(y_true_train, y_pred_train, digits=6)
    print(classification_report_string)

    print("Report on test set:")
    classification_report_string = classification_report(y_true_test, y_pred_test, digits=6)
    print(classification_report_string)

glimpse_df(df)