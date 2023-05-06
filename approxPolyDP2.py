import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import natsort
import math


# List of image file paths to process
image_paths = glob.glob("C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/*.png")
image_paths = natsort.natsorted(image_paths)

# Create an empty dataframe to store the results
df = pd.DataFrame(columns=['Filename', 'C X', 'C Y', 'Above X', 'Above Y', 'Below X', 'Below Y'])
data = []
for image_path in image_paths:
    # Load an image to segment and preprocess it
    image = cv2.imread(image_path, 0)

    # Apply thresholding to create a binary mask
    ret, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find the contour points of the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if there are any contours found
    if len(contours) == 0:
        print("No contours found!")
    else:
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
    
        # Find the two closest points above and below C
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
       
        # Print the results
        print("C point: ", C)
        print("Point above center: ", above)
        print("Point below center: ", below)
        print("Distance between C and the point above: ", c_above_distance)
        print("Distance between C and the point below: ", c_below_distance)

    
        # Display the result
        plt.imshow(image, cmap='gray')
        for point in approx:
            cv2.circle(image, tuple(point[0]), 5, (255, 0, 0), -1)
        cv2.circle(image, tuple(C), 5, (0, 255, 0), -1)
        if above is not None:
            pt = tuple(above[0])
            cv2.rectangle(image, (pt[0]-5, pt[1]-5), (pt[0]+5, pt[1]+5), (0, 255, 0), -1)
        if below is not None:
            pt = tuple(below[0])
            cv2.rectangle(image, (pt[0]-5, pt[1]-5), (pt[0]+5, pt[1]+5), (0, 0, 255), -1)

        plt.imshow(image)
        plt.show()
        

        # Construct the output file path using the input file name
        output_file_path = os.path.join("C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/approxPolyDP", os.path.basename(image_path))
        cv2.imwrite(output_file_path, image)
        
        data.append({'Filename': os.path.basename(image_path),
                     'C X': C[0] if C is not None else 'None',
                     'C Y': C[1] if C is not None else 'None',
                     'Above X': above[0][0] if above is not None else 'None',
                     'Above Y': above[0][1] if above is not None else 'None',
                     'Below X': below[0][0] if below is not None else 'None',
                     'Below Y': below[0][1] if below is not None else 'None'})
        df = pd.DataFrame(data)
        df.to_excel('results.xlsx', index=False)
        
        plt.show()

    

