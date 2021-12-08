import argparse
from itertools import chain, combinations
import warnings
from IPython.core.display import display
from pandas import read_pickle
from pandas._config.config import set_option
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import pickle
from joblib import dump, load
from utils_file_saver import load_model, save_model
from utils_functions import glimpse_df, powerset
from utils_paths import PATH_DATA_MODEL
from itertools import product
from model import model_rfc, model_mlp, model_svc, model_knn
from utils_env import channels_good
from itertools import combinations

"""
get trained model
get dataset
refit model
itterate over the electrode pairs and caculate acc
"""

set_option("display.max_columns", None)

parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("--svm", metavar="file", required=True, type=str, help="SVM model used for caclulating the accuracy")
args = parser.parse_args()


model: SVC = load_model(args.svm).best_estimator_
df = read_pickle(args.df)
df["label"] = df["label"].astype(int)


### Split to X Y train test
X = df.loc[:, ~df.columns.isin(["label"])]
y = df.loc[:, "label"]

X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=0)

### Hyperparameters are optimized in previous fitting
### Fit again to avoid prediction on seen data
model.scoring = "accuracy"

channels_pairs = list(combinations(channels_good, 2))
channel_acc = {}

for e in channels_good:

    X_train = X_train_org.loc[:, X_train_org.columns.str.contains(e)]
    X_test = X_test_org.loc[:, X_test_org.columns.str.contains(e)]

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    channel_acc[e] = accuracy_score(y_test, y_test_pred)

print(channel_acc)
channel_weights = []
for i, channels_i in enumerate(channels_good):
    print("Dual progress", i / len(channels_good) * 100, "%")
    acc_dual = 0
    for channels_j in channels_good:
        if channels_j == channels_i:
            break

        X_train = X_train_org.loc[:, X_train_org.columns.str.contains("|".join([channels_i, channels_j]))]
        X_test = X_test_org.loc[:, X_test_org.columns.str.contains("|".join([channels_i, channels_j]))]

        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        acc_ij = accuracy_score(y_test, y_test_pred)
        acc_dual += acc_ij + channel_acc[channels_i] - channel_acc[channels_j]

    weight = (channel_acc[channels_i] + acc_dual) / len(channels_good)
    channel_weights.append([channels_i, weight])
    print(channels_i, weight)

channel_weights = sorted(channel_weights, key=lambda x: x[1], reverse=True)
print(channel_weights)
