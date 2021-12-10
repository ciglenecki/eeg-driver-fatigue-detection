import argparse
from itertools import chain, combinations
from pathlib import Path
import warnings
from IPython.core.display import display
from pandas import read_pickle
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeavePGroupsOut
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
from joblib import dump, load
from utils_file_saver import save_model
from utils_functions import glimpse_df, powerset, min_max_scaler, stdout_to_file, get_timestamp
from utils_paths import PATH_MODEL, PATH_REPORT
from utils_env import num_users
from itertools import product
from model import model_svc_wide, wide_params, model_svc, model_mlp
from tqdm import tqdm
from utils_env import entropy_channel_combinations
from sklearn.metrics import accuracy_score

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
args = parser.parse_args()
stdout_to_file(Path(PATH_REPORT, "-".join(["svm-parameters", get_timestamp()]) + ".txt"))

df = read_pickle(args.df)
glimpse_df(df)
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, df.columns.isin(["label", "user_id"])]

training_columns = X.columns.isin(entropy_channel_combinations)
groups = X["user_id"].to_numpy()
acc_parameters = []

for C, gamma in tqdm(list(product(wide_params, wide_params))):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    acc_total = 0

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.iloc[train_index, training_columns], X.iloc[test_index, training_columns]
        y_train, y_test = y.iloc[train_index]["label"], y.iloc[test_index]["label"]
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        acc_total += acc
    acc_parameters.append([acc_total / num_users, C, gamma])

print("Acc\t\t\tC\tgamma")
accs = sorted(acc_parameters, key=lambda x: x[0], reverse=True)
for acc in accs:
    print(acc)
