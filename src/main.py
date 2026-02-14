from pathlib import Path

from build_dataframe import build_dataframe
from model import build_feature_matrix, compute_similarity_matrix
from recommender import recommend_with_explanation, find_song_index
from model import compute_pca
from visualize import plot_songs
import pandas as pd


def main():
    data_path = Path.home() / "data" / "small_dataset"

    csv_path = Path("songs.csv")
    
    if csv_path.exists():
        print("Loading songs from CSV...")
        df = pd.read_csv(csv_path, index_col=0)
        print(f"Loaded {len(df)} songs from CSV")
    else:
        # Option B: Build from scratch if CSV doesn't exist
        print("CSV not found, building from dataset...")
        data_path = Path("data/small_dataset")
        df = build_dataframe(data_path, limit=None)  # Load ALL songs
        df = df.dropna().reset_index(drop=True)
        df.to_csv("songs.csv", index=True)
        print(f"Saved {len(df)} songs to CSV")
    
    df = df.dropna().reset_index(drop=True)

    # Let user choose profile
    print("\nChoose recommendation style:")
    print("1. Balanced (all features equal)")
    print("2. Vibe Match (musical feel/energy)")
    print("3. Artist Similar (find similar artists)")
    print("4. Energy Match (tempo/loudness focus)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    profiles = ["balanced", "vibe_match", "artist_similar", "energy_match"]
    profile = profiles[int(choice) - 1] if choice in "1234" else "balanced"
    
    print(f"\nUsing '{profile}' profile")
    
    X_scaled = build_feature_matrix(df, profile=profile)
    similarity_matrix = compute_similarity_matrix(X_scaled)

    # ask user for a song title to recommend similar songs
    query = input("Enter a song title: ")
    idx = find_song_index(df, query)

    if idx is None:
        return

    # show recommendations with explanations
    print("\nRecommended Songs:")
    recommend_with_explanation(idx, df, similarity_matrix, X_scaled, top_n=5)
    
    show_plot = input("\nGenerate visualization? (y/n): ").lower()
    
    if show_plot == 'y':
        similarity_scores = similarity_matrix[idx]
        recommended_indices = similarity_scores.argsort()[-6:-1][::-1]

        # Generate PCA for visualization
        X_pca, _ = compute_pca(X_scaled)
        plot_songs(X_pca, df, highlighted_idx=idx, recommended_indices=recommended_indices)
    
    print("\nDone!")


if __name__ == "__main__":
    main()