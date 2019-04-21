import torch
import os
from train.train_model import predict
import preprocess.utility as sp
from flask import current_app as app

__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Production"


def separate_file(dir_path):
    """
    Separates and saves the music file into vocals and accompaniment.

    Parameters
    ----------
    dir_path: str
        directory path
    """
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

    sp.write(path=os.path.join(dir_path, 'vocals.wav'),
             data=vocals_estimate,
             sr=sr)
    sp.write(path=os.path.join(dir_path, 'accompaniment.wav'),
             data=acc_estimate,
             sr=sr)
