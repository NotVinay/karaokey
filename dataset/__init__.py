import os
import numpy as np


class Dataset(object):
    """

    """

    def __init__(self,
                     dir_path=None,
                     set='train',
                     source_label='vocals',
                     data_type='.npy',
                     lazy_load=True,
                 ):
        self.data_type = data_type
        self.set = set
        self.source_label = source_label
        self.dir_path = os.path.expanduser(dir_path)
        self.lazy_load = lazy_load
        self.tracks = self.load_track_dirs(self.set)

        self.lazy_load = lazy_load

        if not self.lazy_load:
            self._mixtures, self._labels = self.load_all_tracks()

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
        mixture : ndarray, shape(nb_channels, nb_samples)
            audio data of mixture file
        label : ndarray, shape(nb_channels, nb_samples)
            audio data of mixture file

        """
        if self.lazy_load:
            mixture, label = self.load(self.tracks[index])
        else:
            mixture = self._mixtures[index]
            label = self._labels[index]

        return mixture, label

    @property
    def mixtures(self):
        if not self.lazy_load:
            return self._mixtures
        else:
            return None

    @property
    def labels(self):
        if not self.lazy_load:
            return self._labels
        else:
            return None

    def load_track_dirs(self, set="train"):

        set_folder = os.path.join(self.dir_path, set)
        _, folders, _ = next(os.walk(set_folder))

        track_list = sorted(folders)
        track_dirs = []
        for track_name in track_list:
            track_folder = os.path.join(set_folder, track_name)
            track_dirs.append(track_folder)

        return track_dirs

    def load(self, track_dir):
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
        mixtures = []
        labels = []

        for track_dir in self.tracks:
            mixture, label = self.load(track_dir)
            mixtures.append(mixture)
            labels.append(label)
        return np.array(mixtures), np.array(labels)