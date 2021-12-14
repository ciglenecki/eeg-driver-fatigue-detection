import argparse
from zipfile import ZipFile
from utils_paths import *
from pathlib import Path
import os


def unzip_cnt(zip_path: Path, out_dir: Path):
    """
    Unzip CNT dataset files to out_dir
    """
    subdir = Path(out_dir, "cnt")
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(subdir)

    zips = [file for file in subdir.iterdir() if str(file).endswith(".zip")]
    for zip_item in zips:
        if not str(zip_item).endswith(".zip"):
            continue
        zip_ref = ZipFile(zip_item)
        for cnt_file in zip_ref.namelist()[1:]:  # ignore "9/" directory

            prefix_number = zip_item.stem  # 9
            state_name = Path(cnt_file).stem.lower().split(" ")[0]  # "Normal State" -> "normal"
            filename = Path(prefix_number + "_" + state_name + ".cnt")

            with open(Path(subdir, filename), "wb") as new_file:
                new_file.write(zip_ref.read(cnt_file))
    """ Delete tmp zips """
    for zip_item in zips:
        os.remove(zip_item)

    print("Unzipped cnt dataset to:", out_dir)


def unzip_mat(zip_path: Path, out_dir: Path):
    """
    Unzip MAT dataset files to out_dir
    """
    subdir = Path(out_dir, "mat")
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(subdir)
    print("Unzipped mat dataset to:", out_dir)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", metavar="path", type=str, help="Directory where zips will be exported", default=PATH_DATASET)
parser.add_argument("-m", "--mat", metavar="path", type=str, help="Zip with mat files", default=PATH_ZIP_MAT)
parser.add_argument("-c", "--cnt", metavar="path", type=str, help="Zip with cnt files", default=PATH_ZIP_CNT)
args = parser.parse_args()

out_dir = Path(args.dir)
out_dir.mkdir(parents=True, exist_ok=True)
path_cnt = Path(args.cnt)
path_mat = Path(args.mat)

unzip_cnt(path_cnt, out_dir)
unzip_mat(path_mat, out_dir)
