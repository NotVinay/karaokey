import os
import librosa
import numpy as np


class Dataset(object):
    """`MUSMAG Dataset.
    Args:
        root (string): Root directory of dataset where musmag is stored
        train (bool, optional)
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self,
                 dir_path=None,
                 set='train',
                 label='vocals',
                 file_type='.wav',
                 scale=False,
                 lazy_load=False,
                 ):
        self.file_type = file_type
        self.set = set
        self.label = label
        self.dir_path = os.path.expanduser(dir_path)
        self.lazy_load = lazy_load
        self.tracks = self.load_track_dirs(self.set)

        self.lazy_load = lazy_load

        if not self.lazy_load:
            self.mixture, self.label = self._get_data()
            self.mixture = np.array(self.X)
            self.label = np.array(self.Y)

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
            mixture, label = self.load_track_path(self.tracks[index])
        else:
            mixture = self.mixture[index]
            label = self.label[index]


        return mixture, label

    def __repr__(self):
        """
        String representation of dataset

        Returns
        -------
        s: str
            lengths of tracks in dataset
        """
        s = "Mixture: %s\n" % str(((len(self.tracks),)))
        s += "Label: %s" % str(((len(self.tracks),)))
        return s

    def load_track_dirs(self, set="train"):
        if set is not None:
            if isinstance(set, str):
                set = [set]
            else:
                set = set
        else:
            sets = ['train', 'test']

        track_dirs = []
        for set_name in sets:
            set_folder = os.path.join(
                self.root_dir,
                set_name
            )
            _, folders, _ = next(os.walk(set_folder))

            track_list = sorted(folders)

            for track_name in track_list:
                track_folder = os.path.join(set_folder, track_name)
                track_dirs.append(track_folder)
        return track_dirs

    def load_track_path(self, track_dir):
        mixture_path = os.path.join(
            track_dir,
            'mixture' + self.data_type
        )

        mixture = np.load(mixture_path, mmap_mode='c')

        # add track to list of tracks
        label_path = os.path.join(
            track_dir,
            self.label + self.data_type
        )
        label = np.load(label_path, mmap_mode='c')

        return mixture, label

    def _get_data(self):
        mixtures = []
        labels = []

        for track_dir in self.tracks:
            mixture, label = self.load_track_path(track_dir)
            mixtures.append(mixture)
            labels.append(label)

        return mixtures, labels
