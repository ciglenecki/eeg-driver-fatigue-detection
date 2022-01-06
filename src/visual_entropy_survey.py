from pathlib import Path
from typing import Dict
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

from utils_paths import PATH_DATASET_MAT

from utils_feature_extraction import dict_apply_procedture, isnull_values_any
from utils_functions import min_max_dataframe, standard_scale_dataframe

mat = scipy.io.loadmat(Path(PATH_DATASET_MAT, "1.mat"))
keys = [key for key in mat.keys() if not key.startswith("__")]

[print(key, "with shape", mat[key].shape) for key in keys]

is_normal_state_mask = pd.Series(mat["Class_label"].squeeze() == 0)
keys_entropy = ["FE", "SE", "AE", "PE"]

print(is_normal_state_mask)

entropies: Dict[str, pd.DataFrame] = {}
for key in keys_entropy:
    entropies[key] = pd.DataFrame(mat[key])
    print(key, "\n", entropies[key].describe())


fig = plt.figure(figsize=(20, 20), tight_layout={"h_pad": 2})

ax = fig.add_subplot(3, 1, 1)
for name, entropy in entropies.items():
    ax.set_title("Original")
    ax.scatter(entropy.index.tolist(), entropy.loc[:, 1], label=name)
    ax.legend()

entropies_scaled = dict_apply_procedture(entropies, min_max_dataframe)
ax = fig.add_subplot(3, 1, 2)
for name, entropy in entropies_scaled.items():
    ax.set_title("Min_max")
    ax.scatter(entropy.index.tolist(), entropy.loc[:, 1], label=name)
    ax.legend()

entropies_scaled_2 = dict_apply_procedture(entropies, standard_scale_dataframe)
ax = fig.add_subplot(3, 1, 3)
for name, entropy in entropies_scaled_2.items():
    ax.set_title("Standard scaler")
    ax.scatter(entropy.index.tolist(), entropy.loc[:, 1], label=name)
    ax.legend()
plt.show()

# Null and NaN are the same in Pandas :)


for name, entropy in entropies_scaled.items():
    if isnull_values_any(entropy):
        print("Entropy", name, "has null values")
    if isnull_values_any(entropy):
        print("Entropy", name, "has null values")
