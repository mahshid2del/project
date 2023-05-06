import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.measure import find_contours

# Load an image to segment and preprocess it
image = plt.imread('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/frame89_mask.png')

mask = gaussian(image, sigma=1) > 0.5

# Apply thresholding to create a binary mask
threshold = 0.5
binary_mask = mask > threshold

# Find the contour points of the binary mask
contours = find_contours(binary_mask, 0.5)

# Find the center and radius of the semi-circle
center = np.mean(contours[0], axis=0)
radius = np.mean(np.linalg.norm(contours[0] - center, axis=1))
print (center)
# Find the three sharp edges of the semi-circle
edges = []
for point in contours[0]:
    distance = np.linalg.norm(point - center)
    if abs(distance - radius) < 1:
        edges.append(point)
        
# Display the result
fig, ax = plt.subplots()
ax.imshow(image.squeeze(), cmap='gray')
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='r')
ax.scatter(center[1], center[0], color='g', marker='o')
for edge in edges:
    ax.scatter(edge[1], edge[0], color='b', marker='o')
plt.show()
print (edges)
