import numpy as np
import soundfile as sf
import librosa as lb
import os


def read(path=None,
         is_wav=True,
         stereo=True):
    """
    Reads an audio file in time series

    Parameters
    ----------
    path : str
        Path of the audio file
    is_wav : bool
        File format of the file is wav.
        True by default.
    stereo : bool
        Read file as stereo. True by default.

    Returns
    -------
    ndarray
        shape(nb_samples, nb_channels)
    """
    data, sr = None, None

    if not os.path.isfile(path):
        return data, sr

    if is_wav:
        # use soundfile.read for wav files
        data, sr = sf.read(path, always_2d=stereo)
    else:
        # librosa for other compressed files.
        data, sr = lb.load(path, sr=None, mono=not stereo)
        # librosa gives (nb_channels, nb_samples)
        # thus transforming it to (nb_channels, nb_samples)
        data = data.T
    return data, sr


def write(path=None,
          data=None,
          sr=None):
    """
    Writes the music file.

    Parameters
    ----------
    path: str
        path to the file.
    data: ndarray, shape(nb_samples, nb_channels)
        time series data.
    sr: int
        sampling rate
    """
    sf.write(path, data, sr)

def to_mono(data):
    """
    Transforms a stereo track into mono track.

    Parameters
    ----------
    data : ndArray, shape (nb_samples, nb_channels)
        audio signal of `ndim = 2`

    Returns
    -------
    ndarray
        shape=(nb_samples, nb_channels)
        audio signal of `ndim = 1`

    Examples
    --------
    Basic example on how stereo track array is converted to mono

    >>> x = np.array([[2, 7],[8, 5], [9, 3], [5, 1], [2, 6]])
    >>> print(to_mono(data=x))
    [[4.5]
     [6.5]
     [6. ]
     [3. ]
     [4. ]]
    """
    return np.atleast_2d(np.mean(data, axis=1)).T
