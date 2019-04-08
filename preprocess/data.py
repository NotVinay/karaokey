"""
This module helps to load dataset of .wav files.
"""
import os
from preprocess.track_classes import Track

__author__ = "Vinay Patel"
__version__ = "0.1.0"
__maintainer__ = "Vinay Patel"
__email__ = "w1572032@my.westminster.ac.uk"
__status__ = "Development"

class Data:
    """
    Data object to load tracks in .wav format

    Methods
    -------
    get_tracks()
        Iterates through the track folders and loads the tracks as "Track" objects
    """

    def __init__(self, dataset_path):
        """
        Initialises the Data object

        Parameters
        ----------
        dataset_path: str
            Path of the dataset directory
        """
        self.dataset_path = os.path.expanduser(dataset_path)

    def get_tracks(self,
                   sub_set='train',
                   select_tracks=None,
                   labels={'vocals'}):
        """
        loads and returns the tracks from the dataset.

        Parameters
        ----------
        sub_set : str
            The set to load from the dataset
        select_tracks : list[str], optional
            Select the tracks to load.
            If none supplied than all tracks in the directory will be loaded.
        labels : dict
            the labels to load from the dataset.
            "Mixture" will be loaded by default.

        Returns
        -------
        list[Track]
            list of tracks where each track is represents by ''Track'' object
        """
        # list to hold Track objects
        tracks = []

        # geting absolute path of sub_set folder/directory
        set_dir = os.path.join(self.dataset_path, sub_set)

        # getting all folders in set_dir using os.walk
        for _, folders, _ in os.walk(set_dir):
            # iterating through each track directory in sub_set directory
            for track_dir in sorted(folders):
                # skip this iteration if select_tracks are enforced
                # and this track is not present in it
                if (select_tracks is not None) and (track_dir not in select_tracks):
                    continue
                # instantiate and append Track
                track_dir_path = os.path.join(set_dir, track_dir)
                track = Track(title=track_dir,
                              dir_path=track_dir_path,
                              labels=labels)
                tracks.append(track)
                # END OF FOR LOOP of track dir
            # END OF FOR LOOP of set_dir os.walk
        return tracks
