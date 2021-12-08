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
from utils_functions import glimpse_df, powerset, min_max_scaler, stdout_to_file
from utils_paths import PATH_DATA_MODEL, PATH_REPORT
from utils_env import num_users
from itertools import product
from model import model_svc_wide, wide_params
from tqdm import tqdm
from utils_env import entropy_channel_combinations
from sklearn.metrics import accuracy_score
from datetime import datetime


set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
args = parser.parse_args()
timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

report_filename = Path(PATH_REPORT, "-".join(["loo-parameters", timestamp]) + ".txt")
stdout_to_file(report_filename)

df = read_pickle(args.df)
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, df.columns.isin(["label", "user_id"])]

X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.5, random_state=0)

training_columns = X_train_org.columns.isin(entropy_channel_combinations)


acc_parameters = []
for C, gamma in tqdm(list(product(wide_params, wide_params))):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    acc_total = 0
    for user_id in range(num_users):

        X_train = X_train_org.loc[X_train_org["user_id"] != user_id, training_columns]
        X_test = X_test_org.loc[X_test_org["user_id"] != user_id, training_columns]

        y_train = y_train_org[y_train_org["user_id"] != user_id]["label"]
        y_test = y_test_org[y_test_org["user_id"] != user_id]["label"]

        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        acc_total += accuracy_score(y_test, y_test_pred)
    acc_parameters.append([acc_total / num_users, C, gamma])

print("Acc\t\t\tC\tgamma")
accs = sorted(acc_parameters, key=lambda x: x[0], reverse=True)
for acc in accs:
    print(acc)
