"""
Specified paths -- directory structure
"""
from pathlib import Path
from os import getcwd

PATH_CWD = Path(getcwd())
PATH_DATA = Path(PATH_CWD, "data")
PATH_REPORT = Path(PATH_DATA, "reports")
PATH_DATASET = Path(PATH_DATA, "dataset")
PATH_MODEL = Path(PATH_DATA, "models")

PATH_DATAFRAME = Path(PATH_DATA, "dataframes")
PATH_DATASET_MAT = Path(PATH_DATASET, "mat")
PATH_DATASET_CNT = Path(PATH_DATASET, "cnt")

PATH_ZIP_CNT = Path(PATH_DATASET, "5202739.zip")
PATH_ZIP_MAT = Path(PATH_DATASET, "5202751.zip")
