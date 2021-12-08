FREQ = 1000
num_users = 12
EPOCH_SECONDS = 1

signal_offset = -20
SIGNAL_FILE_DURATION_SECONDS = 600
SIGNAL_DURATION_SECONDS_DEFAULT = 300

FATIGUE_STR = "fatigue"
NORMAL_STR = "normal"

elect_all = ["HEOL", "HEOR", "FP1", "FP2", "VEOU", "VEOL", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "A1", "T5", "P3", "PZ", "P4", "T6", "A2", "O1", "OZ", "O2", "FT9", "FT10", "PO1", "PO2"]

elect_good = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]

elect_bad = list(set(elect_all) - set(elect_good))

elect_ignore = []

PAPER_G = 2 ** (-5)
PAPER_C = 2 ** (-1)
PAPER_RFC_TREES = 500
PAPER_RFC_INPUT_VARIABLES = 22
PAPER_BF_HIDDEN = 22


ENTROPIES = ["PE", "AE", "SE", "FE"]

STATES = [NORMAL_STR, FATIGUE_STR]