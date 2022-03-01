"""
Create a summary and comparison for multiple models
"""
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
from sklearn.metrics import (RocCurveDisplay, accuracy_score, f1_score,
                             roc_auc_score)
from tabulate import tabulate

from utils_file_saver import (get_decorated_filepath, get_model_basename,
                              load_model, save_figure)
from utils_functions import get_timestamp, stdout_to_file
from utils_paths import PATH_FIGURE, PATH_REPORT

set_option("display.max_columns", None)
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", metavar="path", required=True, type=str, help="Models which will be summarized")
parser.add_argument("-r", "--output-report", metavar="dir", required=False, type=str, help="Directory where report file will be created.", default=PATH_REPORT)
args = parser.parse_args()
stdout_to_file(Path(args.output_report, "-".join(["model-summary", get_timestamp()]) + ".txt"))

filepaths = glob.glob(args.model)
columns = ["Model name", "F1 score", "Accuracy", "Area under curve"]
table = []

for filepath in filepaths:

    model_pickle = load_model(filepath)
    model, y_trues, y_preds = model_pickle["model"], model_pickle["y_trues"], model_pickle["y_preds"]
    num_of_fits = len(y_trues)
    model_name: str = type(model.estimator).__name__

    """Caculate metrics and add them to the table"""
    accuracies = list(map(lambda x: accuracy_score(x[0], x[1]), zip(y_trues, y_preds)))
    f1s = list(map(lambda x: f1_score(x[0], x[1]), zip(y_trues, y_preds)))
    aucs = list(map(lambda x: roc_auc_score(x[0], x[1]), zip(y_trues, y_preds)))

    accuracy = sum(accuracies) / num_of_fits
    f1 = sum(f1s) / num_of_fits
    auc = sum(aucs) / num_of_fits
    table.append([model_name, f1, accuracy, auc])

    """Print metrics for each driver"""
    for i, (driver_acc, driver_f1, driver_auc) in enumerate(zip(accuracies, f1s, aucs)):
        print("Driver {} F1 {}, Acc {}, AUC {}".format(i, driver_f1, driver_acc, driver_auc))

    """Plot ROC figure and save it"""
    figure_basename = get_model_basename(model_name, auc, name_tag="roc")
    figure_filepath = get_decorated_filepath(
        directory=PATH_FIGURE,
        basename=figure_basename,
        extension="png",
        metadata=model.best_params_,
        datetime_arg=True,
    )
    fig, ax = plt.subplots()
    ax.set_title("{} ROC (area under curve {:.4f})".format(model_name, auc))
    plt.grid()
    for i, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        label = "ROC ({:.4f})".format(auc) if num_of_fits == 1 else "Participant {} ROC ({:.4f})".format(i + 1, aucs[i])
        roc_plot = RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, ax=ax, alpha=0.9, label=label)
    save_figure(figure_filepath, metadata=model.best_params_)

"""Print models' summary"""
table = sorted(table, key=lambda x: x[1], reverse=True)
print(tabulate(table, headers=columns, tablefmt="github"), "\n")

"""Bar plot accuracy, f1 and ROC for each model and save it"""
df = DataFrame(table, columns=columns)
df_numerical = df.iloc[:, 1:]
ax = df.plot(kind="bar", legend="best", width=0.9, figsize=(11, 4))
ax.set_xticklabels(df["Model name"], rotation=0)
ax.set_ylim([df_numerical.min().min() - 0.01, df_numerical.max().max()])

for p in ax.containers:
    ax.bar_label(p, fmt="%.4f", label_type="edge")

figure_filepath = get_decorated_filepath(
    directory=PATH_FIGURE,
    basename="model-compare",
    extension="png",
    datetime_arg=True,
)
save_figure(figure_filepath)
