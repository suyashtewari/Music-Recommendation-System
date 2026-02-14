import os
import h5py
import pandas as pd

def extract_song_features(file_path):
    try:
        with h5py.File(file_path, "r") as f:
            meta = f["metadata"]["songs"][0]
            ana = f["analysis"]["songs"][0]

            return {
                "song_id": meta["song_id"].decode(),
                "title": meta["title"].decode(),
                "artist_name": meta["artist_name"].decode(),
                "release": meta["release"].decode(),
                "artist_familiarity": meta["artist_familiarity"],
                "artist_hotttnesss": meta["artist_hotttnesss"],
                "song_hotttnesss": meta["song_hotttnesss"],
                "tempo": ana["tempo"],
                "loudness": ana["loudness"],
                "duration": ana["duration"],
                "key": ana["key"],
                "mode": ana["mode"],
                "time_signature": ana["time_signature"],
            }
        
    except Exception as e:
        return None
    

def build_dataframe(dataset_folder, limit=100):
    rows = []
    count = 0

    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if not file.endswith(".h5"):
                continue

            path = os.path.join(root, file)
            data = extract_song_features(path)

            if data:
                rows.append(data)
                count += 1

            if limit and count >= limit:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataframe("./data/small_dataset", limit=1200)
    print("Loaded songs:", len(df))
    # df = df.dropna()
    df.to_csv("songs.csv", index=True)
