# Music Recommendation System

A content-based music recommendation system built using the Million Song Dataset. This system analyzes audio features like tempo, loudness, energy, and more to find songs that sound similar to your favorites.

### Prerequisites

```bash
# Required libraries
pip install numpy pandas scikit-learn h5py matplotlib
```

```bash
python build_dataframe.py
```
This creates `songs.csv` with all extracted features

```bash
python main.py
```

## Usage

### Basic Recommendation

```bash
$ python main.py

Enter a song title: wonderwall

Found: Wonderwall by Oasis

================================================================================
Base Song: Wonderwall by Oasis
================================================================================

1. Don't Look Back In Anger by Oasis
   Similarity Score: 0.892
   Most similar features: artist_familiarity, artist_hotttnesss, tempo
   Feature comparison:
      • artist_familiarity: 0.85 vs 0.85
      • artist_hotttnesss: 0.78 vs 0.78
      • tempo: 87.50 vs 89.20

2. Champagne Supernova by Oasis
   Similarity Score: 0.876
   ...

✓ Plot saved as pca_plot.png
```

### Multiple Matches

```bash
Enter a song title: stay

Found 3 matches:
1. Stay by Rihanna
2. Stay With Me by Sam Smith
3. Stay (I Missed You) by Lisa Loeb

Select a song (1-3): 2
```

## Dataset Information
- **Website**: http://millionsongdataset.com/
