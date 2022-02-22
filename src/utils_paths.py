"""
Specified paths -- directory structure
"""
from os import getcwd
from pathlib import Path

PATH_CWD = Path(getcwd())
PATH_DATA = Path(PATH_CWD, "data")
PATH_REPORT = Path(PATH_DATA, "reports")
PATH_FIGURE = Path(PATH_DATA, "figures")
PATH_DATASET = Path(PATH_DATA, "dataset.ignoreme")
PATH_MODEL = Path(PATH_DATA, "models")
PATH_DATAFRAME = Path(PATH_DATA, "dataframes")
PATH_DATASET_MAT = Path(PATH_DATASET, "mat")
PATH_DATASET_CNT = Path(PATH_DATASET, "cnt")

PATH_ZIP_CNT = Path(PATH_DATASET, "5202739.zip")
PATH_ZIP_MAT = Path(PATH_DATASET, "5202751.zip")

if __name__ == "__main__":
    pass
