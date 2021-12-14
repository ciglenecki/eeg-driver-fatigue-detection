num_users = 12
FATIGUE_STR = "fatigue"
NORMAL_STR = "normal"
states = [NORMAL_STR, FATIGUE_STR]

""" Signal config"""
FREQ = 1000
EPOCH_SECONDS = 1
signal_offset = -20
SIGNAL_FILE_DURATION_SECONDS = 600
SIGNAL_DURATION_SECONDS_DEFAULT = 300
NOTCH_FILTER_HZ = 50
LOW_PASS_FILTER_RANGE_HZ = [0.15, 40]

""" Channels config """
channels_all = ["HEOL", "HEOR", "FP1", "FP2", "VEOU", "VEOL", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "A1", "T5", "P3", "PZ", "P4", "T6", "A2", "O1", "OZ", "O2", "FT9", "FT10", "PO1", "PO2"]
channels_good = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]
channels_bad = list(set(channels_all) - set(channels_good))
channels_ignore = []


entropy_names = ["PE", "AE", "SE", "FE"]
entropy_channel_combinations = ["{}_{}".format(entropy, channel) for entropy in entropy_names for channel in channels_good]  # [PE_FP1, PE_FP2, ... , PE_C3, AE_FP1, AE_FP2, ..., FE_C3]

""" Train parameters"""
PAPER_G = 2 ** (-5)
PAPER_C = 2 ** (-1)
PAPER_RFC_TREES = 500
PAPER_RFC_INPUT_VARIABLES = 22
PAPER_BF_HIDDEN = 22
