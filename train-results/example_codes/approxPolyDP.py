# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:24:13 2023

@author: mahsh
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image to segment and preprocess it
image = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/frame89_mask.png', 0)

# Get the size of the image
print( image.shape )

# Apply thresholding to create a binary mask
ret, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find the contour points of the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the vertices of the semi-circle using approxPolyDP
epsilon = 0.01 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)

# Calculate the center coordinates of the cell
M = cv2.moments(contours[0])
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
center = (cx, cy)

# Find the closest point to the center and name it C
distances = np.sqrt((approx[:, 0, 0] - cx) ** 2 + (approx[:, 0, 1] - cy) ** 2)
c_index = np.argmin(distances)
C = approx[c_index][0]
                      
above, below = None, None
for point in approx:
    if point[0, 1] < C[1] and point[0, 0] > C[0] and (above is None or point[0, 1] > above[0, 1]):
        above = point
    elif point[0, 1] > C[1] and point[0, 0] > C[0] and (below is None or point[0, 1] < below[0, 1]):
        below = point

# Print the results
print("Center point: ", C)
print("Point above center: ", above)
print("Point below center: ", below)

# Draw a line between C and above
cv2.line(image, (C[0], C[1]), (above[0][0], above[0][1]), (0, 0, 255), 3)
cv2.line(image, (C[0], C[1]), (below[0][0], below[0][1]), (0, 0, 255), 3)

# Display the result
plt.imshow(image, cmap='gray')
plt.scatter(approx[:, 0, 0], approx[:, 0, 1], color='r', marker='o')
plt.scatter(C[0], C[1], color='g', marker='x')
plt.scatter(above[0][0], above[0][1], color='b', marker='o')
plt.scatter(below[0][0], below[0][1], color='y', marker='o')
plt.show()
