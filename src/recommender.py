def recommend(song_index, df, similarty_matrix, top_n=5):
    similarity_scores = similarty_matrix[song_index]

    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]

    return df.iloc[similar_indices][["title", "artist_name"]]


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
            choice = int(input(f"\nSelect a song (1-{len(matches)}): "))
            if 1 <= choice <= len(matches):
                return matches.index[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(matches)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nSearch cancelled.")
            return None