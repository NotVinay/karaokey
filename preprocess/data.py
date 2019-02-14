import os
from preprocess.track_classes import Track, Audio

class data(object):
    """
        Data object to retrieve tracks
    """

    def __init__(self, dataset_path):
        self.dataset_path = os.path.expanduser(dataset_path)

    def load_tracks(self,
                    set='train',
                    select_tracks=None,
                    labels={'vocals'}):
        """

        Parameters
        ----------
        set : str
            select set of the dataset
        select_tracks : list[Track]
            select specific tracks from the dataset
        labels :
            select the labels of the music tracks

        Returns
        -------
        list[Track]
            list of 'Track'(s)
        """
        tracks = []
        set_folder = os.path.join(self.dataset_path, set)

        for _, folders, _ in os.walk(set_folder):
            # load the tracks using Track object
            for track_dir in sorted(folders):
                if (select_tracks is not None) and (track_dir not in select_tracks):
                    continue
                track_dir_path = os.path.join(set_folder, track_dir)
                track = Track(title=track_dir,
                              dir_path=track_dir_path,
                              labels=labels)
                tracks.append(track)
        return tracks