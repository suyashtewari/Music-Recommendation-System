from pathlib import Path

from build_dataframe import build_dataframe
from model import build_feature_matrix, compute_similarity_matrix
from recommender import recommend, find_song_index


def main():
    data_path = Path(__file__).resolve().parent.parent / "data/small_dataset"

    df = build_dataframe(data_path, limit=200)

    df = df.dropna().reset_index(drop=True)

    X_scaled = build_feature_matrix(df)

    similarity_matrix = compute_similarity_matrix(X_scaled)

    query = input("Enter a song title: ")
    idx = find_song_index(df, query)

    if idx is None:
        return

    print("\nBase Song:")
    print(df.iloc[idx][["title", "artist_name"]])

    print("\nRecommended Songs:")
    recommendations = recommend(idx, df, similarity_matrix)
    print(recommendations)


if __name__ == "__main__":
    main()
