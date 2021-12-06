from pathlib import Path
from os import getcwd

PATH_CWD = Path(getcwd())
PATH_DATA = Path(PATH_CWD, "data")
PATH_DATA_MAT = Path(PATH_DATA, "mat")
PATH_DATA_CNT = Path(PATH_DATA, "cnt")
PATH_DATA_DATAFRAME = Path(PATH_DATA, "dataframes")
PATH_DATA_MODEL = Path(PATH_DATA, "models")

PATH_ZIP_CNT = Path(PATH_DATA, "5202739.zip")
PATH_ZIP_MAT = Path(PATH_DATA, "5202751.zip")
