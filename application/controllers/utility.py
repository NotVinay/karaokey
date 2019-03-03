import numpy as np
import soundfile as sf
import librosa as lb
import os


def read(path=None,
         is_wav=True,
         stereo=True):
    data, sr = None, None

    if not os.path.isfile(path):
        return data, sr

    if is_wav:
        data, sr = sf.read(path, always_2d=stereo)
    else:
        data, sr = lb.load(path, sr=None, mono=not stereo)
        data = data.T
    return data, sr


def to_mono(data):
    """
    Transforms a stereo track into mono track.

    Parameters
    ----------
    data : ndArray, shape (nb_samples, nb_channels)
            audio signal of `ndim = 2`

    Returns
    -------
    ndArray, shape=(nb_frames, nb_channels)
            audio signal of `ndim = 2`
    """
    return np.atleast_2d(np.mean(data, axis=1)).T
