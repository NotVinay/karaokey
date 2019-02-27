from config.model_config import DATASET_CONFIG, PREPROCESS_CONFIG
from preprocess.data import Data
from application.controllers.preprocess_tools import STFT, Scaler
import application.controllers.utility as sp
import numpy as np
import os


if __name__ == '__main__':
    SET = 'train'
    MONO = True
    data = Data(dataset_path=DATASET_CONFIG.PATH)
    tracks = data.load_tracks(set=SET, labels={'vocals', 'accompaniment'})
    set_path = os.path.join(PREPROCESS_CONFIG.PATH, SET)
    if not os.path.exists(PREPROCESS_CONFIG.PATH):
        os.mkdir(PREPROCESS_CONFIG.PATH)

    if not os.path.exists(set_path):
        os.mkdir(set_path)

    for i, track in enumerate(tracks):
        # transformation object
        transform = STFT(sr=DATASET_CONFIG.SR,
                         n_per_seg=DATASET_CONFIG.N_PER_SEG,
                         n_overlap=DATASET_CONFIG.N_OVERLAP)

        # Scaler object
        scaler = Scaler()

        # make dir for saving processed track
        track_dir = os.path.join(set_path, str(track))
        if not os.path.exists(track_dir):
            os.mkdir(track_dir)

        # time series data of mixture
        data_mix = track.mixture.data
        print("mixture: ", data_mix.shape)

        # convert to mono
        if MONO:
            data_mix = sp.to_mono(data_mix)
            print("mixture mono: ", data_mix.shape)

        # generate STFT of time series data
        x_mix_tf = transform.stft(data_mix.T)

        # get spectrogram of STFT i.e., |Xi|
        x_mix_stft = np.abs(x_mix_tf)

        # convert stereo spectrogram to mono
        # x_mix_stft_mono = np.sum(x_mix_stft, axis=-1)

        # scaling the values to 0 to 1
        X_mix = scaler.scale(x_mix_stft)
        print("mix mean", np.mean(X_mix))

        # scaling the values to 0 to 1
        track_boundary = scaler.boundary

        mix_path = os.path.join(track_dir, str(track.mixture) + '.npy')
        # np.save(mix_path, X_mix)


        for label in track.sources:
            # time series data for source
            data_src = track.sources[label].data

            # convert to mono
            if MONO:
                data_src = sp.to_mono(data_src)
            print(data_src.shape)

            # generate STFT of time series data
            x_src_tf = transform.stft(data_src.T)

            # get spectrogram of STFT i.e., |Xi|
            x_src_stft = np.abs(x_src_tf)

            # convert stereo spectrogram to mono
            # x_src_stft_mono = np.sum(x_src_stft, axis=-1)

            # scaling the values to 0 to 1
            X_src = scaler.scale(x_src_stft, track_boundary)

            print("src mean", np.mean(X_src))

            src_path = os.path.join(track_dir, str(label) + '.npy')
            # np.save(src_path, X_src)
        print("---------------------------------------------")
