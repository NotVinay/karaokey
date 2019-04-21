"""
Contains track classes which represents audio tracks.
"""


import os
import preprocess.utility as sp

__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Production"


class Audio:
    """
    Represents an audio i.e, either mixture or sources

    Attributes
    ----------
    name : str
        Audio file name
    path : str
        Absolute path audio file
    _data : ndarray
        shape(nb_samples, nb_channels)
        Data of the audio files in time domain.
    _sr : int
        sampling rate of the audio
    ext : str
        extension of the audio
    """

    def __init__(self,
                 name,
                 path=None,
                 ext=None):
        """
        Initialising the Audio Object

        Parameters
        ----------
        name : str
        path : str
            Absolute path audio file
        ext : str
            extension of the audio
        """
        self.name = name
        self.ext = ext
        self.path = os.path.abspath(path)
        self._data = None
        self._sr = None

    @property
    def data(self):
        """
        Time series data samples of the audio file.

        Returns
        -------
        ndarray
            shape(nb_samples, nb_channels)
            Data of the audio files in time series.

        """
        if self._data is not None:
            # return cached
            return self._data
        else:
            # lazy load the data if not cached
            if os.path.exists(self.path) and os.path.isfile(self.path):
                # reading sound file
                data, sr = sp.read(self.path,
                                   is_wav=True,
                                   stereo=True)
                self._sr = sr
                self._data = data
                return self._data
            else:
                # file doesn't exist error
                self._sr = None
                self._data = None
                raise ValueError("Path : %s is doesn't exist" % self.path)

    @property
    def sr(self):
        """
        Get sample rate of the Audio

        Returns
        -------
        int
            sampling rate of audio in Hertz (Hz)
        """
        # if already loaded
        if self._sr is not None:
            # return cached
            return self._sr
        else:
            # lazy load the data if not cached
            if os.path.exists(self.path):
                # reading sound file
                data, sr = sp.read(self.path,
                                   is_wav=True,
                                   stereo=True)
                self._data = data
                self._sr = sr
                return sr
            else:
                # file doesn't exist error
                self._sr = None
                self._data = None
                raise ValueError("Path : %s doesn't exist or not a valid file" % self.path)

    @property
    def duration(self):
        """
        Get duration of the audio file.

        Returns
        -------
        int
            track duration in seconds
        """
        return self._data.shape[0] / self._sr

    @data.setter
    def data(self, x):
        """
        Sets the time series data of the audio file.

        Parameters
        ----------
        x : ndarray
            shape(nb_samples, nb_channels)
            Data of the audio files in time domain.
        """
        self._data = x

    @sr.setter
    def sr(self, sampling_rate):
        """
        Sets the sample rate of the audio file

        Parameters
        ----------
        sampling_rate : int
            sample rate of the audio signal
        """
        self._sr = sampling_rate

    def __str__(self):
        """
        Returns string representation of `Audio`

        Returns
        -------
        str
            String representation of `Audio`
        """
        return self.name

    def __repr__(self):
        """
        Returns string representation of `Audio`

        Returns
        -------
        str
            String representation of `Audio`
        """
        return self.__str__()


class Track:
    """
    Represents an audio track sample i.e., a composition of mixture and its sources

    Attributes
    ----------
    title : str
        Track title
    dir_path : str
        Track directory path
    _mixture : Audio
        Mixture audio of the track
    _sources : dict[Audio]
        Sources of the track
    """
    def __init__(self,
                 title=None,
                 dir_path=None,
                 labels={'vocals'}):
        """
        Constructor of the Track object

        Parameters
        ----------
        title : str
            title of track
        dir_path : str
            track directory path
        labels : dict
            labels to retrieve from the track
        """
        self.title = title
        self.dir_path = os.path.abspath(dir_path)
        self.labels = labels
        self._mixture = None
        self._sources = None

    @property
    def mixture(self):
        """
        Returns the mixture of the track

        Returns
        -------
        mixture : Audio
            mixture of audio of track
        """
        if self._mixture is not None:
            # return cached
            return self._mixture
        else:
            # lazy load the data if not cached
            mix_path = os.path.join(self.dir_path, 'mixture.wav')
            # load mixture
            if os.path.exists(mix_path):
                self._mixture = Audio(name='mixture',
                                      path=mix_path,
                                      ext=".wav")
                return self._mixture
            else:
                # mixture doesn't exist
                self._mixture = None
                raise ValueError("Mixture Path: % doesn't exist", mix_path)

    @property
    def sources(self):
        """
        Generates the sources of the track

        Returns
        -------
        sources : dict[Audio]
            list of audio sources
        """
        if self._sources is not None:
            # return cached
            return self._sources
        else:
            # lazy load the data if not cached
            self._sources = {}
            # load sources for each label
            for label in self.labels:
                file_path = os.path.join(self.dir_path, label + '.wav')
                if os.path.exists(file_path):
                    self._sources[label] = Audio(name=label,
                                                 path=file_path,
                                                 ext=".wav")
                else:
                    # source doesn't exist
                    self._sources[label] = None
                    raise ValueError("Source Path: % doesn't exist", file_path)
                # END OF FOR LOOP of labels
            return self._sources

    @mixture.setter
    def mixture(self, mix: Audio):
        """
        sets the mixture of track
        Parameters
        ----------
        mix : Audio
            mixture of audio of track
        """
        self._mixture = mix

    @sources.setter
    def sources(self, s):
        """
        Sets the sources of the audio track

        Parameters
        ----------
        s : dict[Audio]
            dict of audio sources
        """
        self._sources = s

    def __str__(self):
        """
        Returns string representation of Track

        Returns
        -------
        str
            String representation of Track
        """
        return self.title

    def __repr__(self):
        """
        Returns string representation of Track

        Returns
        -------
        str
            String representation of Track
        """
        return self.__str__()
