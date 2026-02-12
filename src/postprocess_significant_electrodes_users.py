"""
Caculate significant electrodes for each user

In this method weights for each electrode is caculated for each user. Once all weights are obtained, average weights across all users and all electrodes is caculated, resulting in 30 average weights for each electrode.

In every step, we refit the model with input data `X_train` which contains filtered rows (rows with a certain `user_id`) and filtered columns (columns for `electrode` which the accuracy is being caculated for) 
"""

from typing import Dict, List

from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm


def caculate_mode_drivers(model: SVC, X_train_org: DataFrame, X_test_org: DataFrame, y_train_org: DataFrame, y_test_org: DataFrame, channels_good: list, NUM_USERS: int) -> List:
    """
    Calculate single accuracy for each channel (Acc_i) for each driver.
    """
    driver_channel_acc: Dict[str, Dict[str, float]] = {}

    for driver_id in tqdm(range(NUM_USERS)):
        for ch in channels_good:
            X_train = X_train_org.loc[X_train_org["driver_id"] == driver_id, X_train_org.columns.str.contains(ch)]
            X_test = X_test_org.loc[X_test_org["driver_id"] == driver_id, X_test_org.columns.str.contains(ch)]

            y_train = y_train_org[y_train_org["driver_id"] == driver_id]["is_fatigued"]
            y_test = y_test_org[y_test_org["driver_id"] == driver_id]["is_fatigued"]

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            if driver_id not in driver_channel_acc:
                driver_channel_acc[driver_id] = {}
            driver_channel_acc[driver_id][ch] = accuracy_score(y_test, y_test_pred)

    """
    Calculate weight for each driver for each channel (V_i).

    drivers_channel_weights = [
        { #driver1
            "FP1": 0.9,
            "FP2": 0.3
            ...
        },
        { #driver2
            "FP1": 0.5,
            "FP2": 0.6
            ...
        }
        ...
    ]
    """

    drivers_channel_weights = []
    for driver_id in tqdm(range(NUM_USERS)):
        channel_weights = {}

        for channel_a_name in channels_good:
            sum_elements = []

            for channel_b_name in channels_good:
                """
                Calculate Acc(i,j) and add it to sum expression
                """
                if channel_b_name == channel_a_name:
                    break

                X_train = X_train_org.loc[X_train_org["driver_id"] == driver_id, X_train_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

                X_test = X_test_org.loc[X_test_org["driver_id"] == driver_id, X_test_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

                y_train = y_train_org[y_train_org["driver_id"] == driver_id]["is_fatigued"]
                y_test = y_test_org[y_test_org["driver_id"] == driver_id]["is_fatigued"]

                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                acc_ij = accuracy_score(y_test, y_test_pred)
                sum_elements.append(acc_ij + driver_channel_acc[driver_id][channel_a_name] - driver_channel_acc[driver_id][channel_b_name])

            sum_expression = sum(sum_elements)
            acc_i = driver_channel_acc[driver_id][channel_a_name]
            weight = (acc_i + sum_expression) / len(channels_good)
            channel_weights[channel_a_name] = weight
        drivers_channel_weights.append(channel_weights)

    weights = []
    for channel_i in range(len(channels_good)):
        channel_name = channels_good[channel_i]
        avg_weight = sum(map(lambda x: x[channel_name], drivers_channel_weights)) / len(drivers_channel_weights)
        weights.append([channel_name, avg_weight])

    return sorted(weights, key=lambda x: x[1], reverse=True)
# update

