import os
import h5py
import numpy as np

class SongFeatureDataset:
    def __init__(self, folder, limit=100):
        self.songs = {}
        self._load(folder, limit)

    def _load(self, folder, limit):
        count = 0
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".h5"):
                    path = os.path.join(root, file)
                    try:
                        self._read_file(path)
                        count += 1
                        if count >= limit:
                            return
                    except Exception as e:
                        print(f"Skipping {file}: {e}")

    def _read_file(self, path):
        with h5py.File(path, "r") as f:
            metadata = f["metadata"]["songs"]
            analysis = f["analysis"]["songs"]

            song_id = metadata[0]["song_id"].decode()
            title = metadata[0]["title"].decode()
            artist_name = metadata[0]["artist_name"].decode()

            tempo = analysis[0]["tempo"]
            loudness = analysis[0]["loudness"]
            duration = analysis[0]["duration"]
            danceability = analysis[0]["danceability"]
            energy = analysis[0]["energy"]

            self.songs[song_id] = {
                "title": title,
                "artist": artist_name,
                "features": np.array([tempo, loudness, duration, danceability, energy])
            }

    def all_song_ids(self):
        return list(self.songs.keys())

    def get_song_info(self, song_id):
        return self.songs[song_id]

    def get_features(self, song_id):
        return self.songs[song_id]["features"]
