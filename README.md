# Harmonic-Aware Pairing Detection (HAPD)
HAPD is machine learning model used to detect harmonic similarity between musical audio samples. The model converts audio signals into spectral feature vectors and uses clustering and similarity metrics to group musically related inputs.


## Overview
The system takes a folder of audio samples and transforms each file its harmonic content. The model is able to:

1. Extracts chroma features from each audio file  
2. Averages these features into a 12-dimensional vector  
3. Normalizes the feature space  
4. Applies KMeans clustering to group similar samples  
5. Computes cosine similarity between all samples  
6. Visualizes relationships using a similarity heatmap  
7. Outputs structured cluster assignments and feature vectors


## Feature Extraction

Each audio file is processed using `librosa`:

- Audio is loaded at a fixed sample rate (22050 Hz)
- A chroma STFT representation is computed
- The chroma matrix is averaged over time to produce a 12-dimensional vector

This vector represents the harmonic distribution across the 12 pitch classes in the chromatic scale (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) over the duration of the audio clip.

## Preprocessing

Before clustering:

- Feature vectors are converted into a NumPy array
- Standard scaling is applied using `StandardScaler`

This ensures all chroma dimensions contribute equally to clustering.


## Clustering

The model uses KMeans clustering:

- Number of clusters: 5
- Random state fixed for reproducibility

Each audio sample is assigned a cluster label based on harmonic similarity.


## Similarity Analysis

Cosine similarity is computed between all feature vectors to measure pairwise harmonic similarity.

This produces a similarity matrix, where:
- Values close to 1 indicate strong harmonic similarity between samples
- Values near 0 or negative indicate weak harmonic similarity between samples


## Visualization

A heatmap is generated from the similarity matrix:

- Each axis represents the set of audio samples
- Each cell shows similarity between two samples
- Brighter regions indicate stronger harmonic relationships between samples

The visualization shows clustering structure and harmonic grouping patterns in the dataset.

## Output

The program generates:

### 1. Cluster Assignments
Each file is assigned a cluster label.

### 2. Feature Vectors
Each audio sample is represented as a 12-dimensional harmonic feature vector.

### 3. JSON Export
Results are saved to `results.json` in the format:

```json
{
  "file": "example.wav",
  "vector": [...],
  "cluster": 2
}
