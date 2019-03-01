import numpy as np

#   smallest possible positive float value
#   usefull for avoiding "0" values such is in case of division by zero
EPS = np.finfo(np.float).eps

class PREPROCESS_CONFIG:
    PATH = r"C:\Users\w1572032.INTRANET.000\Desktop\pro_dataset_test"

class DATASET_CONFIG:
    PATH = r"C:\Users\w1572032.INTRANET.000\Desktop\dataset_wav"
    SR = 44100
    N_PER_SEG = 4096
    N_HOP_SIZE = N_PER_SEG/2
    N_OVERLAP = N_PER_SEG - N_HOP_SIZE

class TRAIN_CONFIG:
    NB_SAMPLES = 128
    NB_BINS = 2049
    NB_BATCHES = 16
    HIDDEN_SIZE = NB_SAMPLES*2
    LR = 0.001
    STEP = 10000

