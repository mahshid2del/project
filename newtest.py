# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:35:14 2023

@author: mahsh
"""

import cv2
import numpy as np
import glob
import os
import pandas as pd
from math import sqrt
import math
import natsort
import matplotlib.pyplot as plt



#tip_micropipette
def getLeftMostFromImage(mask_image):
    #ret, binary_image = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, binary_image = cv2.threshold(mask_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw the contours on
    contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    # Draw the contours on the image
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    
    leftmost = (-1,-1)    
    if len(contours) > 0:
        cnt = contours[0]
        # Find minimum enclosing circle
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        # Find leftmost pixel
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        # Draw circle and leftmost pixel on image
        cv2.circle(contour_image,center,radius,(0,255,0),2)
        cv2.circle(contour_image,leftmost,5,(0,0,255),-1)
    
    return leftmost, contour_image

# Set the path to the directory containing the image files
images_path = 'C:/Users/mahsh/OneDrive/Bureau/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612/results'

# Get a list of all image files in the directory
input_files = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
output_dir = 'C:/Users/mahsh/OneDrive/Bureau/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612/leftmosts'

pipetteresults = []   
# Loop through each input mask image file and process it with getLeftMostFromImage function
for file_name in input_files:
    input_path = os.path.join(images_path, file_name)
    output_path = os.path.join(output_dir, file_name)
    # Load the input mask image
    mask_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # Process the image and get the leftmost pixel and contour image
    leftmost, ret_img = getLeftMostFromImage(mask_image)
    print(f"{file_name}: {leftmost}")
    pipetteresults.append([file_name, leftmost[0], leftmost[1]])

    # Save the resulting image with the same name as the input image to the output directory
    cv2.imwrite(output_path, ret_img)

print("Done.")


#cell_location
def cell_location(image):
    # Initialize variables with default values
    Major_axis, Minor_axis, Width, Height, center_x, center_y = 0, 0, 0, 0, 0, 0
    
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold and find contours
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes and ellipses on image
    for contour in contours:
        x, y, Width, Height = cv2.boundingRect(contour)
        center_x = x + Width/2
        center_y = y + Height/2
        cv2.rectangle(image, (x, y), (x+Width, y+Height), (0, 255, 0), 2)
            
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(image, ellipse, (0, 0, 255), 2)

            # Extract minor and major axis of ellipse
            (x, y), (Major_axis, Minor_axis), angle = ellipse
            if Minor_axis > Major_axis:
                Minor_axis, Major_axis = Major_axis, Minor_axis

            cv2.putText(image, f"Major axis: {Major_axis:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"Minor axis: {Minor_axis:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"Width: {Width}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"Height: {Height}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    

                
        else:
            print(f"Skipping ellipse fitting for contour {contour}, as it has less than 5 points")

    return Major_axis, Minor_axis, Width, Height, center_x, center_y

input_folder = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results"
output_folder = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/cell_location"

cellLocation = []
for file_name in os.listdir(input_folder):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        # Read image file
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path)

        # Process image and get cell location
        Major_axis, Minor_axis, Width, Height, center_x, center_y = cell_location(image)
        cellLocation.append((file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y))

        # Save output image
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, image)
        print("Processed:", file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y)

print("Done processing all images.")
###################################################################################################

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
        
input_folder = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results"
output_folder = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/approxPolyDP"

approxPolyDPresults=[]

# create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# list all files in the input folder
files = os.listdir(input_folder)

# loop over the files and process each one
for file_name in files:
    # check if file is an image (you can modify this to suit your specific needs)
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        # construct full file paths for input and output images
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        
        # call the approxPolyDP function on the input image
        C, above, below, c_above_distance, c_below_distance = approxPolyDP(input_path)
        approxPolyDPresults.append((file_name, C, above, below, c_above_distance, c_below_distance))

        
        # if C is None, print a message and skip this image
        if C is None:
            print(f"No contours found in {file_name}")
            continue
        
        # save the output image with the closest point marked
        image = cv2.imread(input_path)
        cv2.circle(image, tuple(C), 5, (0, 0, 255), -1)
        cv2.imwrite(output_path, image)
        
        # do something with the output variables, like print the results
        print(f"Image {file_name}:")
        print(f"  Closest point to center: {C}")
        print(f"  Point above center: {above}")
        print(f"  Point below center: {below}")
        print(f"  Distance to point above center: {c_above_distance}")
        print(f"  Distance to point below center: {c_below_distance}")

##########################################################################################
def calculate_contact(cellLocation, pipetteresults):
    contact_results = []
    
    # extract radius for first cell
    file_name, Minor_axis, *_ = cellLocation[0]
    radius = Minor_axis / 2
    print (file_name, radius)
    
    for cell_result, pipette_result in zip(cellLocation, pipetteresults):
        # extract relevant data
        file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y = cell_result
        file_name, leftmost_x, leftmost_y = pipette_result

        # perform deformation calculation
        distance = ((leftmost_x - center_x)**2 + (leftmost_y - center_y)**2)**0.5
        #print (file_name, distance)
        contact = distance - radius

        # add result to deformation_results list
        if 0 < contact < 2:
            contact_results.append((file_name, contact, "contact"))

    # sort the results based on file_name
    sorted_results = sorted(contact_results, key=lambda x: int(x[0].split('_')[0][5:]))

    # return sorted results
    return sorted_results


results = calculate_contact(cellLocation, pipetteresults)
for result in results:
    print(f"{result[0]}: {result[1]:.2f} - {result[2]}")
  
##################################################################

def calculate_contact(cellLocation, pipetteresults):
    contact_results = []
    
    # extract radius for first cell
    file_name, Minor_axis, *_ = cellLocation[0]
    radius = Minor_axis / 2
    print (file_name, radius)
    
    for cell_result, pipette_result in zip(cellLocation, pipetteresults):
        # extract relevant data
        file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y = cell_result
        file_name, leftmost_x, leftmost_y = pipette_result

        # perform deformation calculation
        distance = ((leftmost_x - center_x)**2 + (leftmost_y - center_y)**2)**0.5
        #print (file_name, distance)
        contact = distance - radius

        # add result to deformation_results list
        if 0 < contact < 2:
            # calculate the coordinates of the contact point
            dx = (leftmost_x - center_x) * contact / distance
            dy = (leftmost_y - center_y) * contact / distance
            contact_x = center_x + dx
            contact_y = center_y + dy
            
            contact_results.append((file_name, contact, "contact", contact_x, contact_y))

    # sort the results based on file_name
    sorted_results = sorted(contact_results, key=lambda x: int(x[0].split('_')[0][5:]))

    # return sorted results
    return sorted_results
    
results = calculate_contact(cellLocation, pipetteresults)
for result in results:
    print(f"{result[0]}: {result[1]:.2f} - {result[2]}, {result[3]}, {result[4]}")    


##################################################################


def deformation(cellLocation, pipetteresults, approxPolyDPresults):
    deformation_results = []
    
    file_name, Minor_axis, *_ = cellLocation[0]
    radius = Minor_axis / 2
    
    for cell_result, pipette_result, approxPolyDP_results in zip(cellLocation, pipetteresults, approxPolyDPresults):
        # extract relevant data
        file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y = cell_result
        file_name, leftmost_x, leftmost_y = pipette_result
        file_name, C_x, C_y, above, below, c_above_distance, c_below_distance = approxPolyDP_results
        
        # calculate distance between center of the cell and the leftmost point of the pipette
        pipette_distance = np.sqrt((center_x - leftmost_x)**2 + (center_y - leftmost_y)**2)

        
        # calculate the distance between the center of the cell and the contact point with the pipette
        contact_distance = np.sqrt((C_x - leftmost_x)**2 + (C_y - leftmost_y)**2)
        
        # check if contact point is within the cell radius and close to the C point
        if contact_distance >= radius or contact_distance - radius >= 2 or contact_distance - radius <= 0:
            continue
        
        # calculate the distance between the contact point and the tip of the pipette
        pipette_contact_distance = np.sqrt((C_x - leftmost_x)**2 + (C_y - leftmost_y)**2)
        
        # add the results to the deformation_results list
        deformation_results.append((file_name, pipette_distance, radius, contact_distance, pipette_contact_distance))
    
    return deformation_results

results = deformation(cellLocation, pipetteresults, approxPolyDPresults)
for result in results:
    print(f"{result[0]}: {result[1]:.2f} - {result[2]}")


######################################################################
import math

def calculate_radius(major_axis, minor_axis):
    # calculate the radius of the cell based on its major and minor axis lengths
    return math.sqrt(major_axis**2 + minor_axis**2) / 2

def deformation(cellLocation, pipetteresults, approxPolyDPresults):
    deformation_results = []
    
    for cell_result, pipette_result, approxPolyDP_results in zip(cellLocation, pipetteresults, approxPolyDPresults):
        # extract relevant data
        file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y = cell_result
        file_name, leftmost_x, leftmost_y = pipette_result
        file_name, C_x, C_y, above, below, c_above_distance, c_below_distance = approxPolyDP_results
        
        # calculate the radius of the cell
        radius = calculate_radius(Major_axis, Minor_axis)
        
        # calculate the distance between the center point of the cell and the tip of the pipette
        center_to_pipette_distance = math.sqrt((center_x - leftmost_x)**2 + (center_y - leftmost_y)**2)
        
        # check if the distance is within the desired range
        if 0 < center_to_pipette_distance - radius < 2:
            # calculate the distance between the contact point and the center point of the cell
            contact_to_center_distance = math.sqrt((leftmost_x - C_x)**2 + (leftmost_y - C_y)**2)
            if contact_to_center_distance < 2:
                # add the result to the list
                deformation_results.append((file_name, center_to_pipette_distance, contact_to_center_distance))
    
    return deformation_results
