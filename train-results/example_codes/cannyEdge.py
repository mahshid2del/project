

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.measure import find_contours
from sklearn.cluster import DBSCAN

# Load an image to segment and preprocess it
image = plt.imread('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/frame89_mask.png')

mask = gaussian(image, sigma=1) > 0.5

# Apply thresholding to create a binary mask
threshold = 0.5
binary_mask = mask > threshold

# Apply Canny edge detection to detect the edges of the binary mask
edges = canny(binary_mask, sigma=1)

# Find the contours of the edges
contours = find_contours(edges, 0.5)

# Use DBSCAN clustering to group the edges that are close to each other
edge_points = np.concatenate(contours)
db = DBSCAN(eps=5, min_samples=3)
db.fit(edge_points)

# Find the clusters that represent the sharp edges
sharp_edge_clusters = []
for label in np.unique(db.labels_):
    if label == -1:
        continue
    cluster = edge_points[db.labels_ == label]
    if len(cluster) < 10:
        continue
    sharp_edge_clusters.append(cluster)

# Find the center and radius of the semi-circle
center = np.mean(sharp_edge_clusters[0], axis=0)
radius = np.mean(np.linalg.norm(sharp_edge_clusters[0] - center, axis=1))

# Find the sharp edges of the semi-circle
sharp_edges = []
for cluster in sharp_edge_clusters:
    distance = np.linalg.norm(cluster - center, axis=1)
    indices = np.where(np.abs(distance - radius) < 1)[0]
    sharp_edges.append(cluster[indices])

# Display the result
fig, ax = plt.subplots()
ax.imshow(image.squeeze(), cmap='gray')
for edge in sharp_edges:
    ax.plot(edge[:, 1], edge[:, 0], linewidth=2, color='r')
plt.show()
