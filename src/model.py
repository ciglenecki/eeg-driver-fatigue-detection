import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

PAPER_G = 2 ** (-5)
PAPER_C = 0.5  # 2 ** (-1)
PAPER_RFC_TREES = 500
PAPER_RFC_INPUT_VARIABLES = 22
PAPER_BF_HIDDEN = 22

wide_params = sorted([*np.logspace(-6, 3, 10), 500])

svm_parameters = [{"gamma": sorted([1e-3, PAPER_G, 1e-2]), "C": sorted([PAPER_C, 100, 500, 1000])}]
svm_parameters_wide = [{"gamma": wide_params, "C": wide_params}]
mlp_parameters = {"alpha": [0.00001, 0.0001, 0.001, 0.05], "learning_rate": ["constant"]}
rfc_parameters = {"n_estimators": [PAPER_RFC_TREES], "max_features": [PAPER_RFC_INPUT_VARIABLES, "auto"]}
knn_parameters = {"weights": ["uniform", "distance"]}

model_rfc = GridSearchCV(RandomForestClassifier(oob_score=True), rfc_parameters, return_train_score=True)
model_mlp = GridSearchCV(MLPClassifier(activation="logistic", hidden_layer_sizes=PAPER_BF_HIDDEN, max_iter=500), mlp_parameters, return_train_score=True)
model_svc = GridSearchCV(SVC(kernel="rbf"), svm_parameters, return_train_score=True)
model_svc_wide = GridSearchCV(SVC(kernel="rbf"), svm_parameters_wide, return_train_score=True)
model_knn = GridSearchCV(KNeighborsClassifier(), knn_parameters, return_train_score=True)

if __name__ == "__main__":
    pass
# update
