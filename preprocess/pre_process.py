"""
Loads the Data from the .wav music files, preprocesses it and saves it as numpy array.
Main Pre processing Steps:

- Loading the individual track in time domain using `Data` object.
- Converting the loaded track to mono.
- Transforming it to frequency domain using `STFT` object which does Short Time Fourier Transformation.
- Normalises it by scaling it between 0 to 1 giving 0 to lowest values and 1 to highest values.
- Repeats the steps 1 to 3 for sources and than for scaling it scales the sources relative to the mixture.
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import common.input_handler as uin
from config import DATASET_CONFIG, PREPROCESS_CONFIG
from preprocess.data import Data
from preprocess.preprocess_tools import STFT, Scaler
import preprocess.utility as sp

__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Production"


def main():
    """
    Main method of pre-processing the data and saving it as numpy files
    """
    # set to preprocess
    sub_set = uin.get_input_str(msg="Enter the set for preprocessing?(train/test)",
                                only_accept=['train', 'test'],
                                error_msg="Please enter a valid subset")

    # process it as mono?
    MONO = uin.get_confirmation(msg="Process track as mono?",
                                error_msg="Please enter a y or n")
    print("choosen set: ", sub_set)
    print("Is mono: ", MONO)

    # load .wav tracks using the data class
    data = Data(dataset_path=DATASET_CONFIG.PATH)
    tracks = data.get_tracks(sub_set=sub_set, labels={'vocals', 'accompaniment'})
    set_path = os.path.join(PREPROCESS_CONFIG.PATH, sub_set)

    # create directories for storing processed dataset
    if not os.path.exists(PREPROCESS_CONFIG.PATH):
        os.mkdir(PREPROCESS_CONFIG.PATH)
    if not os.path.exists(set_path):
        os.mkdir(set_path)

    # create scaler object for cross tracks
    mixture_scaler = StandardScaler()
    sources_scaler = {}

    # iterate over the tracks
    for i, track in enumerate(tracks):
        print("iteration number", i)

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

        # get time series data of mixture
        data_mix = track.mixture.data

        # convert track to mono track
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

        # add track data to cross track scaler computation
        mixture_scaler.partial_fit(np.squeeze(X_mix))

        # repeat the process for individual tracks
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

            # save the scaled spectrogram data as numpy array
            src_path = os.path.join(track_dir, str(label) + '.npy')
            np.save(src_path, X_src)

            # add track data to cross track scaler computation corresponding to this source
            if label in sources_scaler:
                sources_scaler[label].partial_fit(np.squeeze(X_src))
            else:
                # create scaler object if doesn't exist already
                sources_scaler[label] = StandardScaler()
                sources_scaler[label].partial_fit(np.squeeze(X_src))
            # END OF FOR LOOP of source
        # END OF FOR LOOP of the track

    # create directory for storing subset metadata
    metadata_path = os.path.join(PREPROCESS_CONFIG.PATH, sub_set + '_metadata')
    if not os.path.exists(metadata_path):
        os.mkdir(metadata_path)
    # save cross track scaler computation as numpy file
    np.save(os.path.join(metadata_path, 'mixture_scale.npy'), mixture_scaler.scale_)
    np.save(os.path.join(metadata_path, 'mixture_mean.npy'), mixture_scaler.mean_)
    for label in sources_scaler:
        source_scale_path = os.path.join(metadata_path, label + '_scale.npy')
        np.save(source_scale_path, sources_scaler[label].scale_)
        source_mean_path = os.path.join(metadata_path, label + '_mean.npy')
        np.save(source_mean_path, sources_scaler[label].mean_)

if __name__ == '__main__':
    # run main method if this is ran as alone an script
    main()
