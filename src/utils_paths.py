from pathlib import Path
from os import getcwd

PATH_CWD = Path(getcwd())
PATH_DATA = Path(PATH_CWD, "data")
PATH_DATASET = Path(PATH_DATA, "dataset")

PATH_DATA_DATAFRAME = Path(PATH_DATA, "dataframes")
PATH_DATA_MODEL = Path(PATH_DATA, "models")
PATH_DATASET_MAT = Path(PATH_DATASET, "mat")
PATH_DATASET_CNT = Path(PATH_DATASET, "cnt")

PATH_ZIP_CNT = Path(PATH_DATASET, "5202739.zip")
PATH_ZIP_MAT = Path(PATH_DATASET, "5202751.zip")
