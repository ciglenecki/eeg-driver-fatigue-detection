from datetime import datetime
import os
from pathlib import Path
import pickle
from typing import Any, Callable
from joblib import dump, load
import numpy as np
from pandas.core.frame import DataFrame
from utils_functions import dict_to_byte_metadata, dict_to_string
from sklearn.model_selection import GridSearchCV
from pandas import read_pickle

# show additional data in file explorer
METADATA_FIELD_TAGS = "user.xdg.tags"
METADATA_FIELD_COMMENT = "user.xdg.comment"


def load_model(path: Path) -> GridSearchCV:
    return pickle.loads(load(path))


def load_dataframe(path: Path) -> DataFrame:
    return read_pickle(path)


def save_to_file(
    obj: object,
    path: Path,
    file_saver: Callable[[object, str], Any] = lambda obj, filename: dump(obj, filename),
):
    file_saver(obj, str(path))
    print("Saved file:\n", str(path))


def save_to_file_with_metadata(
    obj: object,
    dir: Path,
    basename: str,
    extension: str,
    file_saver: Callable[[object, str], Any] = lambda obj, filename: dump(obj, filename),
    metadata: dict = {},
):
    MAX_FILENAME_LEN = 255
    timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    metadata_str = dict_to_string(metadata)
    metadata_bytes = dict_to_byte_metadata(metadata)
    filename = "-".join([basename, timestamp, metadata_str + extension]).lower()[:MAX_FILENAME_LEN]
    file_path = str(Path(dir, filename))

    save_to_file(obj, file_path, file_saver)
    os.setxattr(file_path, METADATA_FIELD_TAGS, metadata_bytes)


def save_model(model, model_name, score, dir: Path, metadata={}, name_tag=""):
    basename = "{model_name}-{score:.4f}-{name_tag}".format(model_name=model_name, score=score, name_tag=name_tag)
    file_saver = lambda model, filename: dump(pickle.dumps(model), filename)
    save_to_file_with_metadata(model, dir, basename, ".model", file_saver, metadata)


def save_df_to_disk(df: DataFrame, is_complete_train: bool, dir: Path, name_tag: str, metadata={}):
    data_type_str = "complete" if is_complete_train else "partial"
    basename = "-".join([data_type_str, name_tag])
    metadata = {} if is_complete_train else metadata
    file_saver = lambda df, filename: df.to_pickle(str(filename))
    save_to_file_with_metadata(df, dir, basename, ".pkl", file_saver, metadata=metadata)
    return


def save_npy_to_disk(ndarray: np.ndarray, dir: Path, name_tag: str, metadata: dict = {}):
    file_saver = lambda ndarray, filename: np.save(str(filename), ndarray)
    save_to_file_with_metadata(ndarray, dir, name_tag, ".npy", file_saver, metadata)
