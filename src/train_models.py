"""
Load the dataset with the --df argument
Use all entropy features during the training phase
Train each model via grid serach method:
    - SVC (SVM)
    - MLPClassifier (Multi-layer Perceptron classifier)
    - RandomForestClassifier
    - KNeighborsClassifier
Save each model to file (data/models)
Create an report for each model
Create a report file
"""
import argparse
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_pickle
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
from sklearn.cluster import DBSCAN
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from tqdm import tqdm

from model import model_knn, model_mlp, model_rfc, model_svc
from preprocess_preprocess_df import split_and_normalize
from utils_env import training_columns_regex
from utils_file_saver import TIMESTAMP_FORMAT, save_model
from utils_functions import glimpse_df, stdout_to_file
from utils_paths import PATH_MODEL, PATH_REPORT

timestamp = datetime.today()
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
set_option("display.max_columns", None)

parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("-r", "--output-report", metavar="dir", required=False, type=str, help="Directory where report file will be created.", default=PATH_REPORT)
parser.add_argument("-s", "--strategies", nargs="+", choices=["split", "leaveoneout"], default=["split"], help="Strategies that will be used for training models.")
args = parser.parse_args()
stdout_to_file(Path(args.output_report, "-".join(["train-models", timestamp.strftime(TIMESTAMP_FORMAT)]) + ".txt"))
print(vars(args), "\n")


def loo_generator(X, y):
    groups = X["driver_id"].to_numpy()
    scaler = MinMaxScaler()

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.loc[train_index, training_columns], X.loc[test_index, training_columns]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train.loc[:, training_columns] = scaler.fit_transform(X_train.loc[:, training_columns])
        X_test.loc[:, training_columns] = scaler.transform(X_test.loc[:, training_columns])
        yield X_train, X_test, y_train, y_test


def split_generator(X, y):
    X_train, X_test, y_train, y_test = split_and_normalize(X.loc[:, training_columns], y, test_size=0.5, columns_to_scale=training_columns)
    # X_train, X_test = df_filter_columns_by_std(X_train, X_test)
    yield X_train, X_test, y_train, y_test


df: DataFrame = read_pickle(args.df)
training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
X = df.drop("is_fatigued", axis=1)
y = df.loc[:, "is_fatigued"]

strategies = {"leaveoneout": loo_generator, "split": split_generator}
scorings = ["f1"]
models = [model_svc, model_rfc, model_mlp, model_knn]
training_generators = map(lambda strategy_name: (strategy_name, strategies[strategy_name]), args.strategies)

for scoring, model in tqdm(list(product(scorings, models)), desc="Training model"):
    scoring: str
    model: GridSearchCV
    model_name = type(model.estimator).__name__
    model.scoring = scoring

    y_trues = []
    y_preds = []
    means = []
    stds = []
    params_dict = {}
    for (training_generator_name, training_generator) in training_generators:
        for X_train, X_test, y_train, y_test in tqdm(list(training_generator(X, y)), desc="Model {}".format(model_name)):
            model.fit(X_train.values, y_train.values)
            y_true_test, y_pred_test = y_test, model.predict(X_test.values)

            y_trues.append(y_true_test)
            y_preds.append(y_pred_test)

            for mean, std, params in zip(model.cv_results_["mean_test_score"], model.cv_results_["std_test_score"], model.cv_results_["params"]):
                params = frozenset(params.items())
                if params not in params_dict:
                    params_dict[params] = {}
                    params_dict[params]["means"] = []
                    params_dict[params]["stds"] = []
                params_dict[params]["means"].append(mean)
                params_dict[params]["stds"].append(std)
        f1_average = sum((map(lambda x: f1_score(x[0], x[1]), zip(y_trues, y_preds)))) / len(y_trues)
        acc_average = sum((map(lambda x: accuracy_score(x[0], x[1]), zip(y_trues, y_preds)))) / len(y_trues)

        print_table = {"Model": [model_name], "f1": [f1_average], "accuracy": [acc_average]}
        print_table.update({k: [v] for k, v in model.best_params_.items()})
        print(tabulate(print_table, headers="keys"), "\n")

        for params in params_dict.keys():
            params_dict[params]["mean"] = sum(params_dict[params]["means"]) / len(params_dict[params]["means"])
            params_dict[params]["std"] = sum(params_dict[params]["stds"]) / len(params_dict[params]["stds"])

        for params, mean, std in map(lambda x: (x[0], x[1]["mean"], x[1]["std"]), sorted(params_dict.items(), key=lambda x: x[1]["mean"], reverse=True)):
            print("%0.6f (+/-%0.6f) for %r" % (mean, std * 2, dict(params)))

        save_model(
            model=model,
            model_name=model_name,
            score=f1_average,
            directory=PATH_MODEL,
            metadata=model.best_params_,
            name_tag=training_generator_name,
            datetime=timestamp,
            y_trues=y_trues,
            y_preds=y_preds,
        )

glimpse_df(df)
