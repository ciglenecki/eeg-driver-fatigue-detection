import argparse
from pathlib import Path

from pandas import read_pickle
from pandas._config.config import set_option

from utils_functions import (get_timestamp, glimpse_df, isnull_any,
                             stdout_to_file)
from utils_paths import PATH_REPORT

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("--df", metavar="file", required=True, type=str, help="Dataframe file used for training")
parser.add_argument("-r", "--output-report", metavar="dir", required=False, type=str, help="Directory where report file will be created.", default=PATH_REPORT)
args = parser.parse_args()

stdout_to_file(Path(args.output_report, "-".join(["checker", Path(args.df).stem, get_timestamp()]) + ".txt"))
print(vars(args))

df = read_pickle(args.df)

print("Minimum values for columns", sorted(list((df.min()))))
print("Maximum values for columns", sorted(list((df.max()))))
print("Cols with none", df.loc[:, isnull_any(df)])

glimpse_df(df)
# update

