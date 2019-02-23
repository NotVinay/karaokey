import os
import preprocess.utility as sp

class Audio(object):
    """
    Represents an audio file source

    Attributes
    ----------
    name : str
        Audio file name
    path : str
        Absolute path audio file
    data : ndarray shape(nb_samples, nb_channels)
        Data of the audio files in time domain.
    sr : int
        sampling rate of the audio
    ext : str
        extension of the audio
    """

    def __init__(
            self,
            name,
            path=None,
            ext=None
    ):
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
        ndarray: shape(nb_samples, nb_channels)
            Data of the audio files in time domain.

        """
        # if already loaded
        if self._data is not None:
            return self._data
        else:
            # lazy load the data when needed
            if os.path.exists(self.path) and os.path.isfile(self.path):
                data, sr = sp.read(self.path, stereo=True)  # always gives stereo track.
                self._sr = sr
                self._data = data
                return self._data
            else:
                self._sr = None
                self._data = None
                raise ValueError("Path : %s doesn't exist or not a valid file" % self.path)
                return self._data

    @property
    def sr(self):
        """
        Get sample rate of the Audio

        Returns
        -------
        int: sampling rate of audio in Hertz (Hz)
        """
        # if already loaded
        if self._sr is not None:
            return self._sr
        # load audio to set rate
        else:
            if os.path.exists(self.path):
                data, sr = sp.read(self.path, stereo=True)
                self._data = data
                self._sr = sr
                return sr
            else:
                self._sr = None
                self._data = None
                raise ValueError("Path : %s doesn't exist or not a valid file" % self.path)

    @property
    def duration(self):
        """
        Get duration of the audio file

        Returns
        -------
        int : track duration in seconds

        """
        return self._data.shape[0] / self._sr

    @data.setter
    def data(self, x):
        """
        Sets the data of the audio file

        Parameters
        ----------
        ndarray: shape(nb_samples, nb_channels)
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
            sample rate of the audio sigmal
        """
        self._sr = sampling_rate

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

class Track(object):
    """
        represents an audio track

        Attributes
        ----------
        title : str
            Track title

        dir_path : str
            track directory path
        _mixture : Audio
            misture audio of the track
        _sources : dict[Audio]
            sources of the track

    """

    def __init__(self,
                 title=None,
                 dir_path=None,
                 labels={'vocals'}
                 ):
        """
            Constructor of the Track object

            Parameters
            ----------
            title : str
                title of track
            dir_path : str
                track directory path
            labels : dict
                labels of the track

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
            return self._mixture
        else:
            mix_path = os.path.join(self.dir_path, 'mixture.wav')
            if os.path.exists(mix_path):
                self._mixture = Audio(name='mixture',
                                      path=mix_path,
                                      ext=".wav")
                return self._mixture
            else:
                raise ValueError("path for mixture does not exist")

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
            return self._sources
        else:
            self._sources = {}
            for label in self.labels:
                file_path = os.path.join(self.dir_path, label + '.wav')
                if os.path.exists(file_path):
                    self._sources[label] = Audio(name=label,
                                                 path=file_path,
                                                 ext=".wav")
                else:
                    self._sources[label] = None
                    raise ValueError("path for mixture does not exist")
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

    # @labels.setter
    # def labels(self, l):
    #     """
    #     Sets the labels of the Track
    #
    #     Parameters
    #     ----------
    #     l : dict
    #         labels of the track
    #     """
    #     self.labels = l

    def __str__(self):
        return self.title


    def __repr__(self):
        return self.__str__()
