"""
Set of wrapper functions which save and load files
Functions are often used for saving reports, models and dataframes 
"""
import functools
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from pandas import read_pickle
from pandas.core.frame import DataFrame
from sklearn.model_selection import GridSearchCV

from utils_functions import dict_to_byte_metadata, dict_to_string

METADATA_FIELD_TAGS = "user.xdg.tags"
METADATA_FIELD_COMMENT = "user.xdg.comment"
MAX_FILENAME_LEN = 255 - 1
TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"


class ModelPickle(TypedDict):
    model: GridSearchCV
    y_trues: Union[None, List[List[int]]]
    y_preds: Union[None, List[List[int]]]


def default_obj_saver(obj, filepath):
    return dump(pickle.dumps(obj), filepath)


def handle_datetime_to_string(arg: Union[datetime, None, bool], string_format: str):
    """
    Returns arg if arg is datetime
    Returns current timestamp if arg is True
    Returns None if arg is falsy/invalid
    """

    if type(arg) is datetime:
        return arg.strftime(string_format)
    if arg is True:
        return datetime.today().strftime(string_format)
    return None


def get_model_basename(model_name: str, score: float, name_tag: Union[str, None] = None):
    basename_parts = filter(lambda x: type(x) is str and bool(x), [model_name, "{score:.4f}".format(score=score), name_tag])
    return "-".join(basename_parts)


def filesaver_decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        filepath = func(*args, **kwargs)
        print("Saved file:")
        print(str(filepath))
        print()
        return filepath

    return wrapper_decorator


def set_file_metadata(filepath, metadata):
    metadata_bytes = dict_to_byte_metadata(metadata)
    os.setxattr(filepath, METADATA_FIELD_TAGS, metadata_bytes)
    return filepath


def load_model(path: Path) -> ModelPickle:
    return pickle.loads(load(path))


def load_dataframe(path: Path) -> DataFrame:
    return read_pickle(path)


def get_decorated_filepath(directory: Path, basename: str, extension: str = None, datetime_arg: Union[datetime, None, bool] = True, metadata: dict = {}):
    """
    Returns filepath

    directory = /my/path/
    basename = basename
    extension = .model
    datetime_arg = 2022-09-01 <class 'datetime.datetime'>
    metadata = {"accuracy": 73, "model": "net"}
    e.g. /my/path/model-2022-09-01-accuracy=73___method=net.model
    """

    timestamp = handle_datetime_to_string(datetime_arg, TIMESTAMP_FORMAT)
    metadata_str = dict_to_string(metadata)
    extension = "." + extension if type(extension) is str else ""

    filename_parts = list(filter(lambda x: type(x) is str and bool(x), [basename, timestamp, metadata_str + extension]))
    filename = "-".join(filename_parts).lower()[:MAX_FILENAME_LEN]
    print(filename)
    return Path(directory, filename)


@filesaver_decorator
def call_file_saver(filepath: Path, file_saver: Callable[[str], Any]):
    file_saver(str(filepath))
    return filepath


def save_obj(
    obj: object,
    filepath: Path,
    file_saver: Callable[[object, str], Any] = default_obj_saver,
    metadata: dict = {},
):
    call_file_saver(filepath, lambda filepath: file_saver(obj, filepath))
    set_file_metadata(filepath, metadata)
    return filepath


def save_figure(filepath: Path, metadata: dict = {}):
    plt.tight_layout()
    call_file_saver(filepath, lambda filepath: plt.savefig(filepath))
    set_file_metadata(filepath, metadata)
    plt.clf()
    plt.close()
    return filepath


def save_model(
    model,
    model_name: str,
    score: float,
    directory: Path,
    metadata={},
    name_tag=None,
    datetime: Union[datetime, None, bool] = True,
    y_trues: Union[None, List[List[int]]] = None,
    y_preds: Union[None, List[List[int]]] = None,
):

    basename = get_model_basename(model_name, score, name_tag)
    filepath = get_decorated_filepath(
        directory=directory,
        basename=basename,
        extension="model",
        datetime_arg=datetime,
        metadata=metadata,
    )

    model_pickle: ModelPickle = {
        "model": model,
        "y_trues": y_trues,
        "y_preds": y_preds,
    }

    call_file_saver(filepath, lambda filepath: dump(pickle.dumps(model_pickle), filepath))
    set_file_metadata(filepath, metadata)
    return filepath


def save_df(df: DataFrame, is_complete_dataset: bool, directory: Path, name_tag: str, metadata={}):
    data_type_str = "complete" if is_complete_dataset else "partial"
    basename = "-".join([data_type_str, name_tag])
    filepath = get_decorated_filepath(directory=directory, basename=basename, extension="pickle", metadata=metadata)
    call_file_saver(filepath, lambda filepath: DataFrame.to_pickle(df, filepath))
    set_file_metadata(filepath, metadata)
    return filepath


def save_npy(ndarray: np.ndarray, is_complete_dataset: bool, directory: Path, name_tag: str, metadata={}):
    data_type_str = "complete" if is_complete_dataset else "partial"
    basename = "-".join([data_type_str, name_tag])
    filepath = get_decorated_filepath(directory=directory, basename=basename, extension="npy", metadata=metadata)
    call_file_saver(filepath, lambda filepath: np.save(str(filepath), ndarray))
    set_file_metadata(filepath, metadata)
    return filepath


if __name__ == "__main__":
    pass
# update

