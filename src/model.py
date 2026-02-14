import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

feature_columns = [
    "artist_familiarity",
    "artist_hotttnesss",
    "song_hotttnesss",
    "tempo",
    "loudness",
    "duration",
    "key",
    "mode",
    "time_signature",
]

WEIGHT_PROFILES = {
    "balanced": {
        "artist_familiarity": 1.0,
        "artist_hotttnesss": 1.0,
        "song_hotttnesss": 1.0,
        "tempo": 1.0,
        "loudness": 1.0,
        "duration": 1.0,
        "key": 1.0,
        "mode": 1.0,
        "time_signature": 1.0,
    },
    "vibe_match": {  # Focus on musical feel
        "artist_familiarity": 0.3,
        "artist_hotttnesss": 0.3,
        "song_hotttnesss": 0.3,
        "tempo": 3.0,
        "loudness": 2.5,
        "duration": 1.5,
        "key": 2.0,
        "mode": 2.0,
        "time_signature": 1.5,
    },
    "artist_similar": {  # Find similar artists
        "artist_familiarity": 5.0,
        "artist_hotttnesss": 5.0,
        "song_hotttnesss": 3.0,
        "tempo": 0.5,
        "loudness": 0.5,
        "duration": 0.5,
        "key": 0.5,
        "mode": 0.5,
        "time_signature": 0.5,
    },
    "energy_match": {  # Match energy level
        "artist_familiarity": 0.5,
        "artist_hotttnesss": 0.5,
        "song_hotttnesss": 1.0,
        "tempo": 4.0,
        "loudness": 4.0,
        "duration": 0.5,
        "key": 0.5,
        "mode": 1.5,
        "time_signature": 1.0,
    }
}

def build_feature_matrix(df, profile="balanced"):
    scaler = StandardScaler()
    features = df[feature_columns].values
    X_scaled = scaler.fit_transform(features)
    
    # Apply profile weights
    weights = np.array([WEIGHT_PROFILES[profile][col] for col in feature_columns])
    X_scaled = X_scaled * weights
    
    return X_scaled

def compute_similarity_matrix(feature_matrix):
    return cosine_similarity(feature_matrix)


def compute_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca
