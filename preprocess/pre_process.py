from config.model_config import DATASET_CONFIG, PREPROCESS_CONFIG
from preprocess.data import Data
from application.controllers.preprocess_tools import STFT, Scaler
from sklearn.preprocessing import StandardScaler
import application.controllers.utility as sp
import numpy as np
import os
print("working")

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

    mixture_scaler = StandardScaler()

    sources_scaler = {}

    for i, track in enumerate(tracks):
        print(i)
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

        # convert to mono
        if MONO:
            data_mix = sp.to_mono(data_mix)

        # generate STFT of time series data
        x_mix_tf = transform.stft(data_mix.T)

        # get spectrogram of STFT i.e., |Xi|
        x_mix_stft = np.abs(x_mix_tf)

        # scaling the values to 0 to 1
        X_mix = scaler.scale(x_mix_stft)

        # scaling the values to 0 to 1
        track_boundary = scaler.boundary

        mix_path = os.path.join(track_dir, str(track.mixture) + '.npy')
        np.save(mix_path, X_mix)

        # save track boundary
        np.save(os.path.join(track_dir, 'boundary.npy'), track_boundary)

        # add to cross track scaler computation
        mixture_scaler.partial_fit(np.squeeze(X_mix))

        for label in track.sources:
            # time series data for source
            data_src = track.sources[label].data

            # convert to mono
            if MONO:
                data_src = sp.to_mono(data_src)

            # generate STFT of time series data
            x_src_tf = transform.stft(data_src.T)

            # get spectrogram of STFT i.e., |Xi|
            x_src_stft = np.abs(x_src_tf)

            # scaling the values to 0 to 1
            X_src = scaler.scale(x_src_stft, track_boundary)

            src_path = os.path.join(track_dir, str(label) + '.npy')
            np.save(src_path, X_src)

            # add to cross track scaler computation
            if label in sources_scaler:
                sources_scaler[label].partial_fit(np.squeeze(X_src))
            else:
                sources_scaler[label] = StandardScaler()
                sources_scaler[label].partial_fit(np.squeeze(X_src))
            # end of source loop
        # end of the track loop

    # save cross track scaler computation as npy
    metadata_path = os.path.join(PREPROCESS_CONFIG.PATH, SET+'_metadata')
    if not os.path.exists(metadata_path):
        os.mkdir(metadata_path)
    np.save(os.path.join(metadata_path, 'mixture_scaler.npy'), mixture_scaler)
    for label in sources_scaler:
        source_scaler_path = os.path.join(metadata_path, label+'_scaler.npy')
        np.save(source_scaler_path, sources_scaler[label])
