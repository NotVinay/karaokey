import os
import numpy as np
from sklearn.preprocessing import StandardScaler


class Dataset(object):
    """
    Class to load the preprocessed dataset

    Attributes
    ----------
    _mixtures : ndarray
        shape(nb_tracks, nb_frames, nb_bins, nb_channels)
        array of scaled and preprocessed spectrogramic mixture tracks
    _labels : ndarray
        shape(nb_tracks, nb_frames, nb_bins, nb_channels)
        array of scaled and preprocessed spectrogramic label tracks
    """

    def __init__(self,
                 dir_path=None,
                 sub_set='train',
                 source_label='vocals',
                 data_type='.npy',
                 lazy_load=True):
        """
        Constructor that initialises the Dataset

        Parameters
        ----------
        dir_path : str
            path of the dataset
        sub_set : str
            sub set to load from the dataset
        source_label : str
            label of the source file to retrieve
        data_type : str
            data type or extension of files
        lazy_load : bool
            Lazy load the data i.e, load on the fly to reduces the memory limits and loading times
        """
        self.data_type = data_type
        self.sub_set = sub_set
        self.source_label = source_label
        self.dir_path = os.path.expanduser(dir_path)
        self.lazy_load = lazy_load
        self.tracks = self.get_track_dir_paths(self.sub_set)
        self.lazy_load = lazy_load

        # pre load all mixture if not lazy load
        if not self.lazy_load:
            self._mixtures, self._labels = self.load_all_tracks()

        # load metadata
        self.mixture_mean, self.mixture_scale, self.label_mean, self.label_scale = self.load_metadata()

    def __len__(self):
        """
        Length of tracks

        Returns
        -------
        int: length of tracks in dataset
        """
        return len(self.tracks)

    def __getitem__(self, index):
        """
        Get the music from the dataset

        Parameters
        ----------
        index : index of the music file

        Returns
        -------
        mixture : ndarray
            shape(nb_frames, nb_bins, nb_channels)
            audio data of mixture file
        label : ndarray
            shape(nb_frames, nb_bins, nb_channels)
            audio data of mixture file

        """
        if self.lazy_load:
            # load if not lazy loaded
            mixture, label = self.load(self.tracks[index])
        else:
            # load if already loaded
            mixture = self._mixtures[index]
            label = self._labels[index]
        return mixture, label

    @property
    def mixtures(self):
        """
        Gets scaled and preprocessed mixtures for all tracks

        Returns
        -------
        mixtures : ndarray
            shape(nb_tracks, nb_frames, nb_bins, nb_channels)
            array of scaled and preprocessed spectrogramic mixture tracks
        """
        if not self.lazy_load:
            return self._mixtures
        else:
            return None

    @property
    def labels(self):
        """
        Gets scaled and preprocessed labels for all tracks

        Returns
        -------
        labels : ndarray
            shape(nb_tracks, nb_frames, nb_bins, nb_channels)
            array of scaled and preprocessed spectrogramic label tracks
        """
        if not self.lazy_load:
            return self._labels
        else:
            return None

    def get_track_dir_paths(self, sub_set="train"):
        """
        Gets track dir paths

        Parameters
        ----------
        sub_set : str
            sub_set of the dataset i.e., train or test

        Returns
        -------
        list[str]
            list of the absolute paths of the track directories.
        """
        set_folder = os.path.join(self.dir_path, sub_set)
        _, folders, _ = next(os.walk(set_folder))

        track_list = sorted(folders)
        track_dirs = []
        for track_name in track_list:
            track_folder = os.path.join(set_folder, track_name)
            track_dirs.append(track_folder)

        return track_dirs

    def get_track_name(self, index):
        """
        Get name of track sampl

        Returns
        -------
        str:
            Name of the track
        """
        track_name = os.path.basename(self.tracks[index])
        return track_name

    def load_metadata(self):
        """
        loads the metadata(scalers) related to the subset of the dataset

        Returns
        -------
        mixture_mean: ndarray
            mean of mixture sources
        mixture_scale: ndarray
            scaled ndarray of mixture sources
        label_mean: ndarray
            mean of label sources
        label_scale: ndarray
            scaled ndarray of label sources
        """
        # loading metadata metadata
        metadata_path = os.path.join(self.dir_path, self.sub_set + '_metadata')
        # load the scalers if saved in the dataset
        if (os.path.exists(os.path.join(metadata_path, 'mixture_scale.npy')) and
                os.path.exists(os.path.join(metadata_path, 'mixture_mean.npy')) and
                os.path.exists(os.path.join(metadata_path, self.source_label + '_scale.npy')) and
                os.path.exists(os.path.join(metadata_path, self.source_label + '_mean.npy'))):
            mixture_scale = np.load(os.path.join(metadata_path, 'mixture_scale.npy'), mmap_mode="r")
            mixture_mean = np.load(os.path.join(metadata_path, 'mixture_mean.npy'), mmap_mode="r")
            label_scale = np.load(os.path.join(metadata_path, self.source_label + '_scale.npy'), mmap_mode="r")
            label_mean = np.load(os.path.join(metadata_path, self.source_label + '_mean.npy'), mmap_mode="r")
        else:
            # if not saved in the set generate on the fly,
            # could be slow for large datasets
            mixture_scaler = StandardScaler()
            label_scaler = StandardScaler()
            for i in range(len(self)):
                mixture, label = self[i]
                mixture_scaler.partial_fit(np.squeeze(mixture))
                label_scaler.partial_fit(np.squeeze(label))
            mixture_scale = mixture_scaler.scale_
            mixture_mean = mixture_scaler.mean_
            label_scale = label_scaler.scale_
            label_mean = label_scaler.mean_
        return mixture_mean, mixture_scale, label_mean, label_scale

    def load(self, track_dir):
        """
        Loads scaled and preprocessed mixture and label for a single track from specified track dir.

        Parameters
        ----------
        track_dir : str
            path of the track directory where the files files are stored

        Returns
        -------
        mixture : ndarray
            shape(nb_frames, nb_bins, nb_channels)
            scaled and preprocessed spectrogramic mixture track
        label : ndarray
            shape(nb_frames, nb_bins, nb_channels)
            scaled and preprocessed spectrogramic label track

        Raises
        ------
        ValueError
            If `track_dir` doesn't exist.
        """
        # get the mixture data
        mixture_path = os.path.join(track_dir,
                                    'mixture' + self.data_type)
        mixture = np.load(mixture_path, mmap_mode="r")
        # get the source label data
        label_path = os.path.join(track_dir,
                                  self.source_label + self.data_type)
        label = np.load(label_path, mmap_mode="r")
        return mixture, label

    def load_all_tracks(self):
        """
        Loads scaled and preprocessed mixtures and labels for all tracks

        Returns
        -------
        mixtures : ndarray
            shape(nb_tracks, nb_frames, nb_bins, nb_channels)
            array of scaled and preprocessed spectrogramic mixture tracks
        labels : ndarray
            shape(nb_tracks, nb_frames, nb_bins, nb_channels)
            array of scaled and preprocessed spectrogramic label tracks
        """
        mixtures = []
        labels = []
        for track_dir in self.tracks:
            mixture, label = self.load(track_dir)
            mixtures.append(mixture)
            labels.append(label)
            # END OF FOR LOOP of tracks
        return np.array(mixtures), np.array(labels)
