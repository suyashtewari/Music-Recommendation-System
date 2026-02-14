import numpy as np
from model import feature_columns


def recommend(song_index, df, similarity_matrix, top_n=5):
    similarity_scores = similarity_matrix[song_index]
    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
    return df.iloc[similar_indices][["title", "artist_name"]]


def recommend_with_explanation(song_index, df, similarity_matrix, X_scaled, top_n=5):
    
    similarity_scores = similarity_matrix[song_index]
    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
    
    base_song = df.iloc[song_index]
    base_features = X_scaled[song_index]
    
    print(f"\n{'='*80}")
    print(f"Base Song: {base_song['title']} by {base_song['artist_name']}")
    print(f"{'='*80}\n")
    
    for rank, idx in enumerate(similar_indices, 1):
        song = df.iloc[idx]
        rec_features = X_scaled[idx]
        similarity = similarity_scores[idx]
        
        # Calculate feature differences (lower = more similar)
        feature_diff = np.abs(base_features - rec_features)
        
        # Get the 3 most similar features (smallest differences)
        most_similar_idx = feature_diff.argsort()[:3]
        most_similar_features = [feature_columns[i] for i in most_similar_idx]
        
        # Get actual values for comparison
        print(f"{rank}. {song['title']} by {song['artist_name']}")
        print(f"   Similarity Score: {similarity:.3f}")
        print(f"   Most similar features: {', '.join(most_similar_features)}")
        
        # Show specific feature comparisons
        print(f"   Feature comparison:")
        for feat_idx in most_similar_idx:
            feat_name = feature_columns[feat_idx]
            base_val = df.iloc[song_index][feat_name]
            rec_val = song[feat_name]
            print(f"      • {feat_name}: {base_val:.2f} vs {rec_val:.2f}")
        print()


def find_song_index(df, title_query):
    title_query = title_query.strip()
    
    if not title_query:
        print("Please enter a song title.")
        return None
    
    matches = df[df["title"].str.contains(title_query, case=False, na=False, regex=False)]
    
    if matches.empty:
        print(f"No matching song found for '{title_query}'.")
        return None

    # Single match - auto-select
    if len(matches) == 1:
        selected = matches.iloc[0]
        print(f"\nFound: {selected['title']} by {selected['artist_name']}")
        return matches.index[0]
    
    # Multiple matches - let user choose
    print(f"\nFound {len(matches)} matches:")
    for i, (idx, row) in enumerate(matches.iterrows(), 1):
        print(f"{i}. {row['title']} by {row['artist_name']}")
    
    while True:
        try:
            choice = input(f"\nSelect a song (1-{len(matches)}) or 'q' to quit: ")
            
            # Allow user to quit
            if choice.lower() == 'q':
                print("Search cancelled.")
                return None
            
            choice_int = int(choice)
            if 1 <= choice_int <= len(matches):
                return matches.index[choice_int - 1]
            else:
                print(f"Please enter a number between 1 and {len(matches)}")

        except ValueError:
            print("Please enter a valid number or 'q' to quit")
            
        except (KeyboardInterrupt, EOFError):
            print("\nSearch cancelled.")
            return None