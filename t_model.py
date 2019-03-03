from config.model_config import DATASET_CONFIG, PREPROCESS_CONFIG
from application.controllers.preprocess_tools import STFT, Scaler
import application.controllers.utility as sp
from preprocess.utility import read, to_mono
import numpy as np
import os
import torch
import norbert
from train.model import LSTM_Model


if __name__ == '__main__':
    print("asdasdasd")
    SET = 'train'
    MONO = True
    # transformation object
    transform = STFT(sr=DATASET_CONFIG.SR,
                     n_per_seg=DATASET_CONFIG.N_PER_SEG,
                     n_overlap=DATASET_CONFIG.N_OVERLAP)

    # Scaler object
    scaler = Scaler()

    # make dir for saving processed track
    track_dir = r'C:\Users\w1572032.INTRANET.000\Desktop\mixture_short.wav'

    # time series data of mixture
    data, sr = read(track_dir, stereo=True)
    print("mixture: ", data.shape)

    # convert to mono
    if MONO:
        data_mix = to_mono(data)
        print("mixture mono: ", data_mix.shape)

    # generate STFT of time series data
    x_tf = transform.stft(data_mix.T)
    print(x_tf.shape)
    # get spectrogram of STFT i.e., |Xi|
    x_mix_stft = np.abs(x_tf)
    # convert stereo spectrogram to mono
    # x_mix_stft_mono = np.sum(x_mix_stft, axis=-1)

    # scaling the values to 0 to 1
    X_scaled = scaler.scale(x_mix_stft)
    print("mix mean", X_scaled.shape)
    X_scaled = np.transpose(X_scaled, (2, 0, 1))
    print(X_scaled.shape)

    # setting up cuda
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(42)
    if cuda_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load app
    path = '/var/www/karaokey/karaokey/controllers/models/lstm.pt'
    model = torch.load(r'H:\FYP\application\controllers\models\lstm.pt', map_location='cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        Y_hat = model(Xt)

    v = Y_hat[0].cpu().detach().numpy()
    # synthesising the outputs to get the results
    v = np.stack([v, v]).transpose(1, 2, 0)
    x_tf_squezzed = np.squeeze(x_tf)
    x_tf_stereo = np.stack([x_tf_squezzed, x_tf_squezzed]).transpose(1, 2, 0)
    v = v[..., None] ** 2
    V = norbert.residual(v, x_tf_stereo)
    Y = norbert.wiener(np.copy(V), np.copy(x_tf_stereo))
    vocals_hat = transform.istft(Y[..., 0])
    acc_hat = transform.istft(Y[..., 1])