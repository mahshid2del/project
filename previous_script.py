import cv2
import numpy as np


def approxPolyDP(image_path):
    # Load an image to segment and preprocess it
    image = cv2.imread(image_path, 0)

    # Apply thresholding to create a binary mask
    ret, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find the contour points of the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) == 0:
        return None, None, None, None, None

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
            
    # Calculate the distance between C and the point above it
    c_above_distance = np.linalg.norm(C - above[0]) if above is not None else '0'
    # Calculate the distance between C and the point below it
    c_below_distance = np.linalg.norm(C - below[0]) if below is not None else '0'
    

    return C, above, below, c_above_distance, c_below_distance
