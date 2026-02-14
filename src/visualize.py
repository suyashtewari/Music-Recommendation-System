import matplotlib.pyplot as plt

def plot_songs(X_pca, df):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)

    plt.title("Song Embedding (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig("pca_plot.png")
    print("Plot saved as pca_plot.png")
