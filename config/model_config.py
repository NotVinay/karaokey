import numpy as np

#   smallest possible positive float value
#   usefull for avoiding "0" values such is in case of division by zero
EPS = np.finfo(np.float).eps

class PREPROCESS_CONFIG:
    PATH = r"C:\Users\w1572032.INTRANET.001\Desktop\pro_dataset_test"

class DATASET_CONFIG:
    PATH = r"C:\Users\w1572032.INTRANET.001\Desktop\short_dataset_wav"
    SR = 44100
    N_PER_SEG = 4096
    N_HOP_SIZE = N_PER_SEG/2
    N_OVERLAP = N_PER_SEG - N_HOP_SIZE

class TRAIN_CONFIG:
    TRAINED_ON = 'accompaniment'
    NB_SAMPLES = 256
    NB_BINS = 2049
    NB_BATCHES = 16
    HIDDEN_SIZE = NB_SAMPLES*2
    NB_LAYERS = 1
    BIDIRECTIONAL = False
    ACTIVATION_FUNCTION = "relu"
    LR = 0.001
    OPTIMIZER = "adam"
    STEPS = 1000

    def __str__(self):
        str = "\n Selected Configuration ->" +\
              "\n Trained on: ", TRAIN_CONFIG.TRAINED_ON +\
              "\n batches: ", str(TRAIN_CONFIG.NB_BATCHES) +\
              "\n samples per batch: ", str(TRAIN_CONFIG.NB_SAMPLES) +\
              "\n Activation Function: ", TRAIN_CONFIG.ACTIVATION_FUNCTION +\
              "\n Hidden Size: ", str(TRAIN_CONFIG.HIDDEN_SIZE) +\
              "\n Layers: ", str(TRAIN_CONFIG.NB_LAYERS) +\
              "\n BiLSTM: ", str(TRAIN_CONFIG.BIDIRECTIONAL) +\
              "\n Learning Rate: ", str(TRAIN_CONFIG.LR) +\
              "\n Optimizer: ", TRAIN_CONFIG.OPTIMIZER + \
              "\n Steps: ", str(TRAIN_CONFIG.STEPS)

