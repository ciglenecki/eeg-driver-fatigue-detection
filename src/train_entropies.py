import argparse
from itertools import chain, combinations
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
from utils_functions import get_dictionary_leaves, glimpse_df, powerset
from utils_paths import PATH_DATA_MODEL
from itertools import product
from model import model_rfc, model_mlp, model_svc, model_knn

set_option("display.max_columns", None)

parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
args = parser.parse_args()


### Load dataframe
df: DataFrame = read_pickle(args.df)

### Split to X Y train test
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]

X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

entropy_excluded_powerset = list(powerset(entropy_names))[:-1]  # Exclude last element where all entropies are mentioned
models = [model_svc]
scorings = ["accuracy"]
results = []

### Train with SVM (accuracy) for every entropy combination
for i, pair in enumerate(product(scorings, models, entropy_excluded_powerset)):
    scoring, model, entropies_exclude = pair

    (X_train, X_test) = (X_train_org.copy(), X_test_org.copy())

    for entropy in entropies_exclude:
        X_train = X_train.loc[:, ~X_train.columns.str.startswith(entropy)]
        X_test = X_test.loc[:, ~X_test.columns.str.startswith(entropy)]
    model_name = type(model.estimator).__name__
    model.scoring = scoring
    model.fit(X_train, y_train)

    # save_model(model=model, model_name=model_name, score=model.best_score_, directory=PATH_DATA_MODEL, metadata=model.best_params_)

    print("=== Best model {} with accuracy {} and parameters {}\n\n".format(model_name, model.best_score_, model.best_params_))

    print("Grid scores on test set:\n")
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params in sorted(zip(means, stds, model.cv_results_["params"]), key=lambda x: x[0]):
        print("%0.6f (+/-%0.6f) for %r" % (mean, std * 2, params))

    y_true_train, y_pred_train = y_train, model.predict(X_train)
    y_true_test, y_pred_test = y_test, model.predict(X_test)

    classification_report_string = classification_report(y_true_test, y_pred_test, digits=6, output_dict=True)

    results.append([i, list(set(entropy_names) - set(entropies_exclude)), model.best_score_, get_dictionary_leaves(classification_report_string)])

for result in sorted(results, key=lambda x: x[2], reverse=True):
    print(result)
