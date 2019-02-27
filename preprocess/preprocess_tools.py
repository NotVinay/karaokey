import numpy as np
from scipy.signal import stft, istft
#from config.model_config import EPS

EPS = np.finfo(np.float).eps

class STFT(object):
    """
    Short Time Fourier Transform: Transforms time series data into frequency domain

    Attributes
    ----------
    sr : sampling rate
    n_per_seg : window size of STFT
    n_overlap : overlap size
    """

    def __init__(self, sr=44100, n_per_seg=4096, n_overlap=3072):
        """
        Constructor

        Parameters
        ----------
        sr : int
            sampling rate
        n_per_seg : int
            window size of STFT
        n_overlap : int
            overlap size
        """
        self.sr = sr
        self.n_per_seg = n_per_seg
        self.n_overlap = n_overlap
        self.shape = None

    def stft(self, audio_data):
        """
        Generates the STFT of time series data

        Parameters
        ----------
        audio_data : ndarray, shape(nb_channels, nb_samples)
            audio signal in time domain.

        Returns
        -------
        data: ndarray, shape(nb_frames, nb_bins, nb_channels)
            STFT of the signal.
        """
        self.shape = audio_data.shape
        f, t, Zxx = stft(
            audio_data,
            nperseg=self.n_per_seg,
            noverlap=self.n_overlap
        )
        return Zxx.T

    def istft(self, X):
        """

        Parameters
        ----------
        X :  ndarray, shape(nb_frames, nb_bins, nb_channels)
            audio signal in time domain.

        Returns
        -------
        audio_data : ndarray, shape(nb_channels, nb_samples)
            audio signal in time domain.
        """

        t, audio_data = istft(
            Zxx=X.T, fs=self.sr, noverlap=self.n_overlap
        )

        return audio_data



class Scaler(object):
    """Apply log compression the magnitude of stft and normalize to _boundary"""

    def ___init___(self):
        self._boundary = None

    def scale(self, X, boundary=None):
        """
        Scales the magnitude of stft

        Parameters
        ----------
        X : ndarray, shape(nb_frames, nb_bins, nb_channels)
            magnitude of stft
        boundary : ndarray
            _boundary values
        Returns
        -------
        X_scale : ndarray
            log scaler of magnitude of stft
        """
        # make sure there is no zero or neg values
        X = np.log(np.maximum(EPS, X))

        if boundary is None:
            self._boundary = self.calc_boundry(X)
        else:
            self._boundary = boundary

        X = np.clip(X, self._boundary[0], self._boundary[1])

        return (X - self._boundary[0]) / (self._boundary[1] - self._boundary[0])

    def unscale(self, X, boundry=None):
        if boundry is None:
            return np.exp(X * (self._boundary[1] - self._boundary[0]) + self._boundary[0])
        else:
            return np.exp(X * (boundry[1] - boundry[0]) + boundry[0])

    def calc_boundry(self, X, min=40):
        return np.percentile(X, (min, 100))

    @property
    def boundary(self):
        return self._boundary


