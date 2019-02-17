from config.model_config import DATASET_CONFIG, PREPROCESS_CONFIG
from preprocess.data import data
from preprocess.preprocess_tools import STFT, Scaler
import numpy as np
import os


def preprocess(data, transform, scaler, track_boundary=None):
    """
    Preprocess the time series data into a scaled spectrogram representation

    Parameters
    ----------
    data : ndarray, [shape=(n, 1) or (2, n)]


    Returns
    -------

    """
    # generate STFT of time series data
    x_tf = transform.stft(data.T)

    # get spectrogram of STFT i.e., |Xi|**2
    x_stft = np.abs(x_tf)**2

    # convert stereo spectrogram to mono
    x_stft_mono = np.sum(x_stft, axis=-1)

    # scaling the values to 0 to 1
    x_scaled = scaler.scale(x_stft_mono, boundary=track_boundary)

    return x_scaled


if __name__ == '__main__':
    set = 'train'
    data = data(dataset_path=DATASET_CONFIG.PATH)
    tracks = data.load_tracks(set=set, labels={'vocals', 'accompaniment'})
    set_path = os.path.join(PREPROCESS_CONFIG.PATH, set)
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

        # generate STFT of time series data
        x_mix_tf = transform.stft(data_mix.T)

        # get spectrogram of STFT i.e., |Xi|**2
        x_mix_stft = np.abs(x_mix_tf) ** 2

        # convert stereo spectrogram to mono
        x_mix_stft_mono = np.sum(x_mix_stft, axis=-1)

        print(np.max(x_mix_stft_mono))
        print(np.min(x_mix_stft_mono))

        # scaling the values to 0 to 1
        X_mix = scaler.scale(x_mix_stft_mono)

        # scaling the values to 0 to 1
        track_boundary = scaler.boundary
        print(track_boundary)

        mix_path = os.path.join(track_dir, str(track.mixture) + '.npy')
        print(np.max(X_mix))
        np.save(mix_path, X_mix)


        for label in track.sources:
            # time series data for source
            data_src = track.sources[label].data

            # generate STFT of time series data
            x_src_tf = transform.stft(data_src.T)

            # get spectrogram of STFT i.e., |Xi|**2
            x_src_stft = np.abs(x_src_tf) ** 2

            # convert stereo spectrogram to mono
            x_src_stft_mono = np.sum(x_src_stft, axis=-1)

            print(np.max(x_src_stft_mono))
            print(np.min(x_src_stft_mono))

            # scaling the values to 0 to 1
            X_src = scaler.scale(x_src_stft_mono, track_boundary)

            src_path = os.path.join(track_dir, str(label) + '.npy')
            print(np.max(X_src))
            np.save(src_path, X_src)
