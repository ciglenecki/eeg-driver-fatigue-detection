import argparse
from itertools import chain, combinations
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
from utils_file_saver import load_model, save_model
from utils_functions import glimpse_df, powerset
from utils_paths import PATH_DATA_MODEL
from itertools import product
from model import model_rfc, model_mlp, model_svc, model_knn


"""
get trained model
get dataset
refit model
itterate over the electrode pairs and caculate acc
"""

set_option("display.max_columns", None)

parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("--model", metavar="file", required=True, type=str, help="Model used for caclulating the accuracy")
args = parser.parse_args()


model: SVC = load_model(args.model).best_estimator_
df = read_pickle(args.df)
df["label"] = df["label"].astype(int)


### Split to X Y train test
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
### Hyperparameters are optimized in previous fitting
model.fit(X_train, y_train)

scorings = ["accuracy"]

for pair in product(scorings, models):
    scoring, model = pair

    model_name = type(model.estimator).__name__
    model.scoring = scoring
    model.fit(X_train, y_train)
    save_model(model=model, model_name=model_name, score=model.best_score_, directory=PATH_DATA_MODEL, metadata=model.best_params_)

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
