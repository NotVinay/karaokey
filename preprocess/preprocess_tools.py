"""
This module contains the tools used for pre-processing music tracks to be feed into the neural network.
"""
import numpy as np
from scipy.signal import stft, istft
from config import EPS

__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Development"


class STFT:
    """
    Short Time Fourier Transform(STFT)
        Transforms time series data into frequency domain

    Attributes
    ----------
    sr : int
        sampling rate
    n_per_seg : int
        window size for computing STFT
    n_overlap : int
        overlap samples for each window shift
    """

    def __init__(self, sr=44100, n_per_seg=4096, n_overlap=3072):
        """
        Constructor

        Parameters
        ----------
        sr : int
            sampling rate
        n_per_seg : int
            window size for computing STFT
        n_overlap : int
            overlap samples for each window shift
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
        audio_data : ndarray
            shape(nb_channels, nb_samples)
            audio signal in time domain.

        Returns
        -------
        data: ndarray
            shape(nb_frames, nb_bins, nb_channels)
            STFT of the signal.

        Notes
        -----
        STFT of signal :math:`x(m)` is given by:

        .. math:: x(t) = x(m)h(m - tH)

        Where the parameters represents following:

        * :math:`x(m)` is signal
        * :math:`t` is time
        * :math:`h(m)` is window
        * :math:`H` = `n_per_seg` - `n_overlap` is hop size of the signal

        See Also
        --------
        istft: inverse STFT (Short Time Fourier Transformation)
        """
        self.shape = audio_data.shape
        f, t, Zxx = stft(audio_data,
                         nperseg=self.n_per_seg,
                         noverlap=self.n_overlap)
        return Zxx.T

    def istft(self, X):
        """
        Generates the inverse STFT of frequency domain data.

        Parameters
        ----------
        X :  ndarray, shape(nb_frames, nb_bins, nb_channels)
            audio signal in time domain.

        Returns
        -------
        data : ndarray, shape(nb_channels, nb_samples)
            audio signal in time domain.

        Notes
        -----
        inverse STFT is given by:

        .. math::
            x(m) = \\frac{ \\sum_{t}x_{t}(m)h(h-tH) }{ \\sum_{t} h^{2}(m-tH) }

        Where the parameters represents following:

        * :math:`x(m)` is retrieved signal
        * :math:`h` is window
        * :math:`t` is time of the signal
        * :math:`H` = `n_per_seg` - `n_overlap` is hop size of the signal

        See Also
        --------
        stft: STFT (Short Time Fourier Transformation)
        """

        t, data = istft(Zxx=X.T,
                        fs=self.sr,
                        noverlap=self.n_overlap)
        return data


class Scaler:
    """
    Apply log compression the magnitude of stft and normalize to `_boundary`

    Attributes
    ----------
    _boundary: ndarray
        Boundary of Log compressed spectrogram
    """

    def ___init___(self):
        """
        initialises Scaler Object
        """
        self._boundary = None

    def scale(self, X, boundary=None):
        """
        Scales the magnitude of STFT(Spectrogram)

        Firstly, it clips the spectrogramic data to percentile bounds
        to reduce the impact of noise in order to get unconsistent result.

        Lastly, it normalises or scales the clipped data to values between 0 and 1.

        Parameters
        ----------
        X : ndarray
            magnitude of stft
        boundary : ndarray
            boundary to which the data needs to be scaled

        Returns
        -------
        X_scale : ndarray
            log scaler of magnitude of stft

        Examples
        --------
        Simple example for calculating scaling for each values.
        Here values lower than 4.2 is scaled to 0 because it's percentile
        boundary was [4.2 9.]

        >>> ex = np.array([[2, 7, 8, 5, 9],[3, 5, 1, 2, 6]])
        >>> scaler = Scaler()
        >>> print(scaler.scale(X=ex))
        [[0.         0.68273064 0.851306   0.25795466 1.        ]
         [0.         0.25795466 0.         0.         0.48812467]]

        See Also
        --------
        unscale: It unscales the scaled spectrogramic mixture i.e. performs reverse process of `scale`.
        """
        # make sure there is no zero or neg values
        X = np.log(np.maximum(EPS, X))

        # take supplied boundary if supplied
        # or compute on the fly
        if boundary is None:
            self._boundary = self.calc_boundary(X)
        else:
            self._boundary = boundary

        """clipping the values to boundary
        Noise in music spectrogram is usually distributed equally,
        Clipping ensures than the all the tf bins have common lowest value possible 
        which can later be eliminated by scaling.
        Thus this helps eliminating noise upto an extent to get the best result.
        """
        X = np.clip(X, self._boundary[0], self._boundary[1])

        return (X - self._boundary[0]) / (self._boundary[1] - self._boundary[0])

    def unscale(self, X, boundary=None):
        """
        It unscales the scaled spectrogramic mixture.
        It performs the reverse process of scaling

        Parameters
        ----------
        X: ndarray
            Scaled spectrogramic data
        boundary :
            boundary to which data was scaled.

        Returns
        -------
        ndarray
            unscaled spectrogramic mixture

        Examples
        --------
        Simple example for unscaling.

        >>> ex = np.array([[0, 0.68273064, 0.851306, 0.25795466, 1], [0, 0.25795466, 0, 0, 0.48812467]])
        >>> scaler = Scaler()
        >>> print(scaler.unscale(X=ex, boundary=np.array([4.2, 9])))
        [[0.         0.68273064 0.851306   0.25795466 1.        ]
         [0.         0.25795466 0.         0.         0.48812467]]

        See Also
        --------
        scale: Scales the magnitude of STFT(Spectrogram).
        """
        if boundary is None:
            return np.exp(X * (self._boundary[1] - self._boundary[0]) + self._boundary[0])
        else:
            print(X * (boundary[1] - boundary[0]) + boundary[0])
            return np.exp(X * (boundary[1] - boundary[0]) + boundary[0])

    def calc_boundary(self, X, min=40):
        """
        Calculates the percentile based boundary values on an ndarray.

        Examples
        --------
        Simple example for calculating boundary.

        >>> ex = np.array([[2, 7, 8, 5, 9],[3, 5, 1, 2, 6]])
        >>> scaler = Scaler()
        >>> print(scaler.calc_boundary(X=ex, min=40))
        [4.2 9. ]


        Parameters
        ----------
        X : ndarray
            input values
        min : int
            min percentile value

        Returns
        -------
        ndarray
            boundary percentiles of value `min` to 100
        """
        return np.percentile(X, (min, 100))

    @property
    def boundary(self):
        """
        Gets the boundary of spectrogramic data.

        Returns
        -------
        ndarray
            boundary of spectrogramic data
        """
        return self._boundary


