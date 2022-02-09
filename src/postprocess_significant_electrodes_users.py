from typing import Dict, List
from pandas.core.frame import DataFrame
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.svm import SVC


def caculate_mode_users(model: SVC, X_train_org: DataFrame, X_test_org: DataFrame, y_train_org: DataFrame, y_test_org: DataFrame, channels_good: list, NUM_USERS: int) -> List:
    """
    Calculate single accuracy for each channel (Acc_i) for each user.
    """
    user_channel_acc: Dict[str, Dict[str, float]] = {}

    for user_id in tqdm(range(NUM_USERS)):
        for ch in channels_good:
            X_train = X_train_org.loc[X_train_org["user_id"] == user_id, X_train_org.columns.str.contains(ch)]
            X_test = X_test_org.loc[X_test_org["user_id"] == user_id, X_test_org.columns.str.contains(ch)]

            y_train = y_train_org[y_train_org["user_id"] == user_id]["is_fatigued"]
            y_test = y_test_org[y_test_org["user_id"] == user_id]["is_fatigued"]

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            if user_id not in user_channel_acc:
                user_channel_acc[user_id] = {}
            user_channel_acc[user_id][ch] = accuracy_score(y_test, y_test_pred)

    """
    Calculate weight for each user for each channel (V_i).

    users_channel_weights = [
        { #user1
            "FP1": 0.9,
            "FP2": 0.3
            ...
        },
        { #user2
            "FP1": 0.5,
            "FP2": 0.6
            ...
        }
        ...
    ]
    """

    users_channel_weights = []
    for user_id in tqdm(range(NUM_USERS)):
        channel_weights = {}

        for channel_a_name in channels_good:
            sum_elements = []

            for channel_b_name in channels_good:
                """
                Calculate Acc(i,j) and add it to sum expression
                """
                if channel_b_name == channel_a_name:
                    break

                X_train = X_train_org.loc[X_train_org["user_id"] == user_id, X_train_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

                X_test = X_test_org.loc[X_test_org["user_id"] == user_id, X_test_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

                y_train = y_train_org[y_train_org["user_id"] == user_id]["is_fatigued"]
                y_test = y_test_org[y_test_org["user_id"] == user_id]["is_fatigued"]

                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                acc_ij = accuracy_score(y_test, y_test_pred)
                sum_elements.append(acc_ij + user_channel_acc[user_id][channel_a_name] - user_channel_acc[user_id][channel_b_name])

            sum_expression = sum(sum_elements)
            acc_i = user_channel_acc[user_id][channel_a_name]
            weight = (acc_i + sum_expression) / len(channels_good)
            channel_weights[channel_a_name] = weight
        users_channel_weights.append(channel_weights)

    weights = []
    for channel_i in range(len(channels_good)):
        channel_name = channels_good[channel_i]
        avg_weight = sum(map(lambda x: x[channel_name], users_channel_weights)) / len(users_channel_weights)
        weights.append([channel_name, avg_weight])

    return sorted(weights, key=lambda x: x[1], reverse=True)
