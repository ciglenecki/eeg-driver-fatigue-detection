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
from utils_env import PAPER_BF_HIDDEN, PAPER_C, PAPER_G, PAPER_RFC_INPUT_VARIABLES, PAPER_RFC_TREES
from utils_file_saver import save_model
from sklearn.neural_network import MLPClassifier
from utils_functions import glimpse_df, powerset
from utils_paths import PATH_DATA_MODEL
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.neighbors import KNeighborsClassifier

set_option("display.max_columns", None)

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--df", metavar="df", required=True, type=str, help="Dataframe file used for training")
args = parser.parse_args()


### Load dataframe
df = read_pickle(args.df)
df["label"] = df["label"].astype(int)
glimpse_df(df)

### Split to X Y train test
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# scorings = ["accuracy", "f1"]
scorings = ["accuracy"]


"""
'bootstrap', 'ccp_alpha', 'class_weight', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'max_samples', 'min_impurity_decrease', 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start']
"""
### Grid parameteres
svm_parameters = [{"gamma": sorted([1e-3, 1e-4, 1e-5, PAPER_G]), "C": sorted([PAPER_C, 1, 100, 500, 1000, 1500])}]
mlp_parameters = {"alpha": [0.00001, 0.0001, 0.001, 0.05], "learning_rate": ["constant", "adaptive"]}
rfc_parameters = {"n_estimators": [PAPER_RFC_TREES], "max_features": [PAPER_RFC_INPUT_VARIABLES, "auto"]}
knn_parameters = {"weights": ["uniform", "distance"]}

### Grid models, the parameters in constructors are static
model_rfc = GridSearchCV(RandomForestClassifier(), rfc_parameters)
model_mlp = GridSearchCV(MLPClassifier(activation="logistic", hidden_layer_sizes=PAPER_BF_HIDDEN, max_iter=500), mlp_parameters)
model_svc = GridSearchCV(SVC(kernel="rbf"), svm_parameters)
model_knn = GridSearchCV(KNeighborsClassifier(), knn_parameters)
models = [model_rfc, model_mlp, model_knn, model_svc]

for pair in product(scorings, models):
    scoring, model = pair

    model_name = type(model.estimator).__name__
    model.scoring = scoring
    model.fit(X_train, y_train)
    save_model(model=model, model_name=model_name, score=model.best_score_, directory=PATH_DATA_MODEL, metadata=model.best_params_)

    print("=== Best model {} with accuracy {} and parameters {}".format(model_name, model.best_score_, model.best_params_))
    print("Grid scores on test set:")
    print()
    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]
    for mean, std, params in sorted(zip(means, stds, model.cv_results_["params"]), key=lambda x: x[0]):
        print("%0.6f (+/-%0.6f) for %r" % (mean, std * 2, params))

    y_true_train, y_pred_train = y_train, model.predict(X_train)
    y_true_test, y_pred_test = y_test, model.predict(X_test)

    print("Report on train set:")
    classification_report_string = classification_report(y_true_train, y_pred_train, digits=6)
    print(classification_report_string)

    print("Report on test set:")
    classification_report_string = classification_report(y_true_test, y_pred_test, digits=6)
    print(classification_report_string)
