from sklearn.model_selection import GridSearchCV, LeaveOneOut, RandomizedSearchCV
from sklearn.svm import SVC
from utils_env import PAPER_BF_HIDDEN, PAPER_C, PAPER_G, PAPER_RFC_INPUT_VARIABLES, PAPER_RFC_TREES
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import accumulate
import numpy as np

wide_params = sorted([*np.logspace(-6, 3, 10), 500])  # from 1e-6 to 1000, mutiplying by ten

svm_parameters = [{"gamma": sorted([1e-3, 1e-4, PAPER_G]), "C": sorted([PAPER_C, 100, 500, 1000, 1500])}]
svm_parameters_wide = [{"gamma": wide_params, "C": wide_params}]
mlp_parameters = {"alpha": [0.00001, 0.0001, 0.001, 0.05], "learning_rate": ["constant", "adaptive"]}
rfc_parameters = {"n_estimators": [PAPER_RFC_TREES], "max_features": [PAPER_RFC_INPUT_VARIABLES, "auto"]}
knn_parameters = {"weights": ["uniform", "distance"]}

model_rfc = GridSearchCV(RandomForestClassifier(), rfc_parameters)
model_mlp = GridSearchCV(MLPClassifier(activation="logistic", hidden_layer_sizes=PAPER_BF_HIDDEN, max_iter=500), mlp_parameters)
model_svc = GridSearchCV(SVC(kernel="rbf"), svm_parameters)
model_svc_wide = GridSearchCV(SVC(kernel="rbf"), svm_parameters_wide)
model_knn = GridSearchCV(KNeighborsClassifier(), knn_parameters)
