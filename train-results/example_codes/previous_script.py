import cv2
import numpy as np


def approxPolyDP(image_path):
    # Load an image to segment and preprocess it
    image = cv2.imread(image_path, 0)

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

    # Calculate the equation of the line between C and above
    slopeH1 = (C[1] - above[0][1]) / (C[0] - above[0][0])
    y_interceptH1 = C[1] - slopeH1 * C[0]

    # Extend the line to the edge of the image
    x1H1 = 0
    y1H1 = int(slopeH1 * x1H1 + y_interceptH1)
    x2H1 = image.shape[1]
    y2H1 = int(slopeH1 * x2H1 + y_interceptH1)
    
    
    # Calculate the equation of the line between C and below
    slopeH2 = (C[1] - below[0][1]) / (C[0] - below[0][0])
    y_interceptH2 = C[1] - slopeH2 * C[0]

    # Extend the line to the edge of the image
    x1H2 = 0
    y1H2 = int(slopeH2 * x1H2 + y_interceptH2)
    x2H2 = image.shape[1]
    y2H2 = int(slopeH2 * x2H2 + y_interceptH2)

    # Return 
    return C, above, below, x1H1, y1H1, x2H1, y2H1 , x1H2, y1H2, x2H2, y2H2

