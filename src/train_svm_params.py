"""
Finds the best C and gamma hyperparameters for SVM model by using Leave One Group out approach.

A single group of rows is defined by participant's id (user_id).
Effectively, this is LOO approach where 1 participant is left for validation and other 11 are used for training the model

Load the dataset with the --df argument
Calculate the accuracy for each hyperparameter pair
"""
import argparse
from itertools import product
from pathlib import Path

from pandas import read_pickle
from pandas._config.config import set_option
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from tqdm import tqdm

from model import wide_params
from utils_env import NUM_USERS, training_columns_regex
from utils_functions import get_timestamp, glimpse_df, stdout_to_file
from utils_paths import PATH_REPORT

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("-r", "--output-report", metavar="dir", required=False, type=str, help="Directory where report file will be created.", default=PATH_REPORT)
args = parser.parse_args()
stdout_to_file(Path(args.output_report, "-".join(["svm-parameters", get_timestamp()]) + ".txt"))

df = read_pickle(args.df)
glimpse_df(df)

X = df.loc[:, ~df.columns.isin(["is_fatigued"])]
y = df.loc[:, df.columns.isin(["is_fatigued", "user_id"])]

training_columns = X.columns.str.contains(training_columns_regex)

groups = X["user_id"].to_numpy()
acc_parameters = []

for C, gamma in tqdm(list(product(wide_params, wide_params))):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    acc_total = 0

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.iloc[train_index, training_columns], X.iloc[test_index, training_columns]
        y_train, y_test = y.iloc[train_index]["is_fatigued"], y.iloc[test_index]["is_fatigued"]
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        acc_total += acc
    acc_parameters.append([acc_total / NUM_USERS, C, gamma])

print("Acc\t\t\tC\tgamma")
accs = sorted(acc_parameters, key=lambda x: x[0], reverse=True)
for acc in accs:
    print(acc)
