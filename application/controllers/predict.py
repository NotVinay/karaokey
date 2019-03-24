import torch
import numpy as np
import norbert
import os
import soundfile as sf
from train.train_model import predict
import preprocess.utility as sp
from preprocess.preprocess_tools import Scaler, STFT
from flask import current_app as app
from train.model import LSTM_Model

from torch.nn import Module, LSTM, Linear, Parameter
import torch.nn.functional as F


def separate_file(dir_path):

    # transformation object
    file_path = os.path.join(dir_path, 'mixture.wav')
    data, sr = sp.read(file_path, stereo=True)

    # setting up cuda
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load app
    dnn_model = torch.load(app.config['MODEL_PATH'], map_location='cpu')
    dnn_model.to(device)
    dnn_model.eval()
    # transformation object
    file_path = os.path.join(dir_path, 'mixture.wav')
    data, sr = sp.read(file_path, stereo=True)
    acc_estimate, vocals_estimate = predict(dnn_model,
                                            device,
                                            data=data,
                                            sr=sr,
                                            trained_on="accompaniment")

    sf.write(os.path.join(dir_path, 'vocals.wav'), vocals_estimate, sr)
    sf.write(os.path.join(dir_path, 'accompaniment.wav'), acc_estimate, sr)