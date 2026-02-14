import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA



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

def build_feature_matrix(df):
    scaler = StandardScaler()
    features = df[feature_columns].values
    return scaler.fit_transform(features)

def compute_similarity_matrix(feature_matrix):
    return cosine_similarity(feature_matrix)


def compute_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca
