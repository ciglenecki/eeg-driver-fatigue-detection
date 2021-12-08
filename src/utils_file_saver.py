from datetime import datetime
import os
from pathlib import Path
import pickle
from joblib import dump, load
from pandas.core.frame import DataFrame
from utils_functions import dict_to_byte_metadata, dict_to_string
from sklearn.model_selection import GridSearchCV

# show additional data in file explorer
METADATA_FIELD_TAGS = "user.xdg.tags"
METADATA_FIELD_COMMENT = "user.xdg.comment"


def save_model(model, model_name, score, directory: Path, metadata={}, prefix=""):
    timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    metadata_string = dict_to_string(metadata)

    name = "{model_name}_{score:.4f}_{prefix}_{timestamp}_{metadata}".format(model_name=model_name, prefix=prefix, score=score, timestamp=timestamp, metadata=metadata_string).lower()

    filename = Path(directory, name)
    dump(pickle.dumps(model), filename)
    os.setxattr(filename, METADATA_FIELD_TAGS, dict_to_byte_metadata(metadata))

    print("Model {} with score {} was saved.".format(model_name, score))
    print(filename)


def load_model(path: Path) -> GridSearchCV:
    return pickle.loads(load(path))


def save_df_to_disk(df: DataFrame, metadata: dict, dir: Path, name: str):

    timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    metadata_str = dict_to_string(metadata)
    metadata_bytes = dict_to_byte_metadata(metadata)

    df_filename = "-".join([name, timestamp, metadata_str + ".pkl"])

    file_path = Path(dir, df_filename)
    df.to_pickle(str(file_path))
    os.setxattr(file_path, METADATA_FIELD_TAGS, metadata_bytes)

    print("Saved file pickle file:", str(file_path))
