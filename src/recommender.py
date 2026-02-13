def recommend(song_index, df, similarty_matrix, top_n=5):
    similarity_scores = similarty_matrix[song_index]

    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]

    return df.iloc[similar_indices][["title", "artist_name"]]

def find_song_index(df, title_query):
    matches = df[df["title"].str.contains(title_query, case=False, na=False)]

    if matches.empty:
        print("No matching song found.")
        return None

    # Show matches so user can confirm
    print("\nMatches found:")
    print(matches[["title", "artist_name"]].head())

    return matches.index[0]
