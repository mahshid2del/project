import cv2
import numpy as np
import matplotlib.pyplot as plt
from previous_script import approxPolyDP
import pandas as pd
from scipy.optimize import fsolve
from math import sqrt
import glob
import natsort
import math


# Define a function to calculate the radius for a given image
def calculate_radius(image_path):
    # Load an image to draw the line on
    cell_image = cv2.imread(image_path)

    # Call the previous function to get the semicircle vertices
    C, above, below, c_above_distance, c_below_distance = approxPolyDP(image_path)

    # Read the deformation values from the Excel file
    merged_df = pd.read_excel('deformation.xlsx')
    rows = merged_df[merged_df['file_name'] == image_path.split('/')[-1]]

    # Check if the DataFrame has any rows before accessing the first row
    if len(rows) > 0:
        deformation_perFrame = float(rows.iloc[0]['deformation_perFrame'])
        print (deformation_perFrame)

    else:
        deformation_perFrame = 0
        print (deformation_perFrame)
    # Calculate the radius
    if c_above_distance is not None and c_below_distance is not None:
        r = sqrt((deformation_perFrame/2)**2 + ((float(c_above_distance) + float(c_below_distance))/2)**2)
    else:
        r = 0
        
    return r , deformation_perFrame

# Define a list of image paths
image_paths = glob.glob('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/*.png')
image_paths = natsort.natsorted(image_paths)

# Calculate the radius for each image in the list
radii = []
for image_path in image_paths:
    r = calculate_radius(image_path)
    radii.append(r)
    print(f"Image: {image_path}, radius: {str(r)}")
    
# Add the radii to the existing Excel file as a new column
merged_df = pd.read_excel('deformation.xlsx').drop_duplicates(subset='file_name', keep='first')
merged_df['dimple_radius'] = radii
merged_df.to_excel('deformation.xlsx', index=False)

