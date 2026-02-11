from feature_loader import SongFeatureDataset

dataset = SongFeatureDataset("small_dataset", limit=50)

songs = dataset.all_song_ids()
print("Loaded songs:", len(songs))

song_id = songs[0]
info = dataset.get_song_info(song_id)
print("First song:", info["title"], "by", info["artist"])
print("Features:", info["features"])
