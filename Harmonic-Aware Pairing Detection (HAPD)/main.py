import librosa
import numpy as np
import os
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Store extracted features
features = []
filenames = []

# Folder containing audio
folder = "samples"

# Loops through audio files
for file in os.listdir(folder):

    if file.endswith(".wav"):

        path = os.path.join(folder, file)

        print("Loading:", file)

        # Load audio
        y, sr = librosa.load(path, sr=22050, mono=True)

        # Extract chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Average chroma over time
        feature_vector = np.mean(chroma, axis=1)

        # Save features
        features.append(feature_vector)
        filenames.append(file)

# Converts to numpy array
features = np.array(features)

print("\nFeature Shape:")
print(features.shape)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("Running clustering on:", len(features_scaled), "samples")

# Creates clustering model
model = KMeans(n_clusters=5, random_state=42)

# Trains model
model.fit(features_scaled)

import json

output = []

print("\nFeature Vectors and Cluster Assignments (saved to file)")

for file, vector, label in zip(filenames, features_scaled, model.labels_):

    entry = {
        "file": file,
        "vector": np.round(vector, 3).tolist(),
        "cluster": int(label)
    }

    output.append(entry)

# write to file
with open("results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved to results.json")

print("\nFeature Vectors and Cluster Assignments:")

for file, vector, label in zip(filenames, features_scaled, model.labels_):

    rounded_vector = np.round(vector, 3)

    print(f"\n{file}")
    print("Vector:", rounded_vector.tolist())
    print("Cluster:", label)

print("Clustering complete")

clusters = defaultdict(list)

for file, label in zip(filenames, model.labels_):
    clusters[label].append(file)

print("\nGrouped Clusters:")
for cluster_id, files in clusters.items():
    print("\nCluster", cluster_id)
    for f in files:
        print(" -", f)

# Similarity matrix
similarity_matrix = cosine_similarity(features_scaled)

print("\nSimilarity Matrix:")
print(similarity_matrix)

def get_most_similar(target_index, top_k=5):
    scores = similarity_matrix[target_index]
    
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\nMost similar to:", filenames[target_index])
    
    for i in range(1, top_k+1):
        idx = sorted_indices[i]
        print(f"{filenames[idx]} -> {scores[idx]:.3f}")

fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)

im = ax.imshow(
    similarity_matrix,
    origin="lower",
    interpolation="nearest"
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Cosine Similarity")

ax.set_title("Harmonic Similarity Matrix of Audio Samples")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Sample Index")

plt.show()

get_most_similar(0)
