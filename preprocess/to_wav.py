"""
This module extracts the sources from the stem files and saves it as wav.

Dependencies needed:
libsndfile and ffmpeg are needed as root dependencies for working with stem files.
stempeg file is needed for python interface.
"""

# import sys
# sys.path.append('../')
import stempeg
import os
import numpy as np
import preprocess.utility as sp


def convert_to_wav(dir_path,
                   wav_dir_path):
    """
    Converts the compressed stem file to individual wav files for mixtures and its sources

    Parameters
    ----------
    dir_path: str
        Path of the stems directory
    wav_dir_path: str
        Path to store the wav files
    """
    if not os.path.exists(wav_dir_path):
        os.mkdir(wav_dir_path)
    # iterating throgh sets
    for sub_set in ['test', 'train']:
        sub_dir = os.path.join(dir_path, sub_set)
        wav_sub_dir = os.path.join(wav_dir_path, sub_set)
        if not os.path.exists(wav_sub_dir):
            os.mkdir(wav_sub_dir)
        # file itterator
        _, folders, files = next(os.walk(sub_dir))
        print("Working with subset ", sub_dir)

        # iterating through files
        for i, f in enumerate(files):
            title = f.split(".stem.mp4")[0].replace("&", "_").replace("'", "_")
            print(i, ": " + title)
            wav_track_path = os.path.join(wav_sub_dir, title)

            if not os.path.exists(wav_track_path):
                os.mkdir(wav_track_path)

            # reading stems
            track_path = os.path.join(sub_dir, f)
            stems, sr = stempeg.read_stems(track_path)

            # saving wav files
            sp.write(path=os.path.join(wav_track_path, "mixture.wav"), data=stems[0], sr=sr)
            sp.write(path=os.path.join(wav_track_path, "vocals.wav"), data=stems[4], sr=sr)
            # generating accompaniment
            acc_list = []
            for source in [1, 2, 3]:
                if stems[source] is not None:
                    acc_list.append(stems[source])
            accompaniment = np.sum(np.array(acc_list), axis=0)
            sp.write(path=os.path.join(wav_track_path, "accompaniment.wav"), data=accompaniment, sr=sr)

if __name__ == '__main__':
    # run main method if this is ran as a script
    convert_to_wav(dir_path=r"E:\Final Year Project\Sample Dataset\short_musdb18_short",
         wav_dir_path=r"C:\Users\w1572032\Desktop\short_wav")
