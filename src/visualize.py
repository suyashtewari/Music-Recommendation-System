import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_songs(X_pca, df, highlighted_idx=None, recommended_indices=None):
    # Delete old plot if it exists
    if os.path.exists("pca_plot.png"):
        os.remove("pca_plot.png")
        print("Removed old plot")
    
    plt.figure(figsize=(14, 10))
    
    # Plot all songs in light gray
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=50, c='lightgray', label='Other Songs')
    
    # Highlight recommended songs in green
    if recommended_indices is not None and len(recommended_indices) > 0:
        plt.scatter(
            X_pca[recommended_indices, 0], 
            X_pca[recommended_indices, 1],
            alpha=0.8, s=150, c='green', 
            edgecolors='darkgreen', linewidth=2,
            label='Recommended Songs'
        )
        
        # Add labels for recommended songs
        for idx in recommended_indices:
            plt.annotate(
                df.iloc[idx]['title'][:20],
                xy=(X_pca[idx, 0], X_pca[idx, 1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.3)
            )
    
    # Highlight selected song in red
    if highlighted_idx is not None:
        plt.scatter(
            X_pca[highlighted_idx, 0], 
            X_pca[highlighted_idx, 1],
            alpha=1.0, s=300, c='red', 
            edgecolors='darkred', linewidth=3,
            marker='*', label='Selected Song'
        )
        
        # Add label for selected song
        plt.annotate(
            df.iloc[highlighted_idx]['title'],
            xy=(X_pca[highlighted_idx, 0], X_pca[highlighted_idx, 1]),
            xytext=(15, 15), textcoords='offset points',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red')
        )
        
        # Draw lines from selected to recommended
        if recommended_indices is not None:
            for rec_idx in recommended_indices:
                plt.plot(
                    [X_pca[highlighted_idx, 0], X_pca[rec_idx, 0]],
                    [X_pca[highlighted_idx, 1], X_pca[rec_idx, 1]],
                    'r--', alpha=0.3, linewidth=1
                )
    
    plt.title("Song Recommendation Space (PCA Projection)", fontsize=16, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("Saving plot...")
    plt.savefig("pca_plot.png", dpi=300, bbox_inches='tight')
    plt.close('all')  # Close all figures
    print("✓ Plot saved as pca_plot.png")