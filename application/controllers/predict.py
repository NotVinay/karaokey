import torch
import numpy as np
import norbert
import os
import soundfile as sf
from .utility import read, to_mono
from .preprocess_tools import Scaler, STFT
from flask import current_app as app
from .model import LSTM_Model

from torch.nn import Module, LSTM, Linear, Parameter
import torch.nn.functional as F


def predict(dir_path):
    global Vanilla

    # transformation object
    file_path = os.path.join(dir_path, 'mixture.wav')
    data, sr = read(file_path, stereo=True)

    transform = STFT(sr=sr,
                     n_per_seg=4096,
                     n_overlap=2048)

    # Scaler object
    scaler = Scaler()

    nb_samples, nb_channels = data.shape

    # change to mono
    if nb_channels > 1:
        data = to_mono(data)

    # generate STFT of time series data
    x_tf = transform.stft(data.T)
    # get spectrogram of STFT i.e., |Xi|
    x_mix_stft = np.abs(x_tf)

    # scaling the values to 0 to 1
    X_scaled = scaler.scale(x_mix_stft)

    X_scaled = np.transpose(X_scaled, (2, 0, 1))

    # setting up cuda
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(42)
    if cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # load app
    model = torch.load(app.config['MODEL_PATH'], map_location='cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        Y_hat = model(Xt)

    v = Y_hat[0].cpu().detach().numpy()
    # synthesising the outputs to get the results
    v = np.stack([v, v]).transpose(1, 2, 0)
    x_tf_squeeze = np.squeeze(x_tf)
    x_tf_stereo = np.stack([x_tf_squeeze, x_tf_squeeze]).transpose(1, 2, 0)
    v = v[..., None] ** 2
    V = norbert.residual(v, x_tf_stereo)
    Y = norbert.wiener(np.copy(V), np.copy(x_tf_stereo))
    vocals_hat = transform.istft(Y[..., 0])
    acc_hat = transform.istft(Y[..., 1])

    vocals_path = os.path.join(dir_path, 'vocals.wav')
    acc_path = os.path.join(dir_path, 'accompaniment.wav')

    sf.write(vocals_path, vocals_hat.T, sr)
    sf.write(acc_path, acc_hat.T, sr)