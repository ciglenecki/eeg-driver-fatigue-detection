"""
Utility functions
"""
import argparse
import re
import sys
from datetime import datetime
from itertools import chain, combinations
from os import getcwd
from pathlib import Path
from typing import Dict, TypeVar

from IPython.display import display
from pandas import DataFrame

T = TypeVar("T")


def serialize_functions(*rest):
    current = rest[len(rest) - 1]
    rest = rest[:-1]
    return lambda x: current(serialize_functions(*rest)(x) if rest else x)


def is_arg_default(arg_name: str, parser: argparse.ArgumentParser, args: argparse.Namespace):
    return parser.get_default("arg_name") == vars(args)[arg_name]


def get_timestamp():
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def dict_apply_procedture(old_dict: Dict[str, T], procedure) -> Dict[str, T]:
    return {k: procedure(v) for k, v in old_dict.items()}


def isnull_any(df):
    # Null and NaN are the same in Pandas :)
    return df.isnull().any()


def isnull_values_sum(df):
    return df.isnull().values.sum() > 0


def isnull_sum(df):
    return df.isnull().sum() > 0


def isnull_values_any(df):
    return df.isnull().values.any()


def rows_with_null(df):
    return df[df.isnull().any(axis=1)]


def get_tmin_tmax(start, duration, end_cutoff):
    return (start - end_cutoff, start + duration - end_cutoff)


def to_numpy_reshape(x):
    return DataFrame.to_numpy(x).reshape(-1, 1)


def get_cnt_filename(i_driver: int, state: str):
    return "{i_driver}_{state}.cnt".format(i_driver=i_driver, state=state)


def glimpse_df(df: DataFrame):

    print("\nShowing first 3 data points\n")
    display(df.head(n=3))

    print("\nShowing last 3 data points\n")
    display(df.tail(n=3))

    print("\nShowing 3 radnom data points\n")
    display(df.sample(n=3))

    display(df.describe())


def powerset(iterable):
    "[1,2,3] --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))


def get_dictionary_leaves(dictionary: dict):
    """
    {a: 3, b: {c: 3, d: 4}} --> [[a,3], [c,3], [d,4]]
    """

    def get_leaves(pair):
        key, value = pair
        if type(value) is dict:
            return get_dictionary_leaves(value)
        return [[key, value]]

    result = []
    for pair in dictionary.items():
        result.extend(get_leaves(pair))
    return result


def dict_to_byte_metadata(dictionary: dict):
    """
    {a:3, b:2, c:test} ---> "a 3, b 2, c test"
    """
    pairs = get_dictionary_leaves(dictionary)
    return ",".join(map(lambda key_value: " ".join([str(key_value[0]), str(key_value[1])]), pairs)).encode()


def dict_to_string(dictionary: dict):
    """
    {"accuracy": 73, "method": "net"} ---> "accuracy_73___method_net"
    """
    pairs = get_dictionary_leaves(dictionary)
    if pairs is None:
        return ""
    return "___".join(map(lambda key_value: "_".join([str(key_value[0]), str(key_value[1])]), pairs))


class SocketConcatenator(object):
    """
    Class adds an abillity to write to mulitple sockets
    For example, write both to stdout and to a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def stdout_to_file(file: Path):
    """
    Pipes standard input to standard input and to a file.
    """
    print()
    print("Standard output piped to file:")
    print(file)
    print()
    f = open(Path(getcwd(), file), "w")
    sys.stdout = SocketConcatenator(sys.stdout, f)


def print_report(file: Path, regex_filter):
    regex = re.compile(regex_filter)
    readlines = open(file, "r").readlines()
    filtered_lines = [s for s in readlines if regex.match(s)]
    for line in filtered_lines:
        print(line.strip())


def flatten(t):
    return [item for sublist in t for item in sublist]


if __name__ == "__main__":
    pass
