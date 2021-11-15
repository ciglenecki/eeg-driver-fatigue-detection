
def dict_apply_procedture(old_dict: Dict[str, T], procedure) -> Dict[str, T]:
    return {k: procedure(v) for k, v in old_dict.items()}


def min_max_dataframe(df: DataFrame):
    return DataFrame(min_max_scaler(df))


def standard_scale_dataframe(df: DataFrame):
    return DataFrame(standard_scaler(df))


standard_scaler = preprocessing.StandardScaler().fit_transform


def standard_scaler_1d(x): return standard_scaler(
    x.reshape(-1, 1)).reshape(1, -1).squeeze()


min_max_scaler = preprocessing.MinMaxScaler().fit_transform


def min_max_scaler_1d(x): return min_max_scaler(
    x.reshape(-1, 1)).reshape(1, -1).squeeze()
# Null and NaN are the same in Pandas :)


def isnull_any(df):
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


def to_numpy_reshape(x): return DataFrame.to_numpy(x).reshape(-1, 1)
