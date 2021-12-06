from zipfile import ZipFile
from utils_paths import *
from pathlib import Path
import os


def unzip_data():
    unzip_cnt()
    unzip_mat()


def unzip_cnt():
    with ZipFile(PATH_ZIP_CNT, "r") as zip_ref:
        zip_ref.extractall(PATH_DATASET_CNT)

    zips = [file for file in PATH_DATASET_CNT.iterdir() if str(file).endswith(".zip")]
    for zip_item in zips:
        if not str(zip_item).endswith(".zip"):
            continue
        zip_ref = ZipFile(zip_item)  # create zipfile object
        for cnt_file in zip_ref.namelist()[1:]:  # ignore "9/" directory

            prefix_number = zip_item.stem  # 9
            state_name = Path(cnt_file).stem.lower().split(" ")[0]  # "Normal State" -> "normal"
            filename = Path(prefix_number + "_" + state_name + ".cnt")

            with open(Path(PATH_DATASET_CNT, filename), "wb") as new_file:
                new_file.write(zip_ref.read(cnt_file))
    # Delete zips as they were temporary
    for zip_item in zips:
        os.remove(zip_item)


def unzip_mat():
    with ZipFile(PATH_ZIP_MAT, "r") as zip_ref:
        zip_ref.extractall(PATH_DATASET_MAT)
