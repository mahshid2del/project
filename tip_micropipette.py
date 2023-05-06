import cv2
import numpy as np
import glob
import os
import csv
import pandas as pd

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
output_file = "C:/Users/mahsh/OneDrive/Bureau/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612/leftmosts/leftmost_pixels.xlsx"

pipetteresults = []   
# Loop through each input mask image file and process it with getLeftMostFromImage function
for file in input_files:
    input_path = os.path.join(images_path, file)
    output_path = os.path.join(output_dir, file)
    # Load the input mask image
    mask_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # Process the image and get the leftmost pixel and contour image
    leftmost, ret_img = getLeftMostFromImage(mask_image)
    print(f"{file}: {leftmost}")
    pipetteresults.append([file, leftmost[0], leftmost[1]])

    # Save the resulting image with the same name as the input image to the output directory
    cv2.imwrite(output_path, ret_img)

print("Done.")

# Save results to Excel file
df = pd.DataFrame(pipetteresults, columns=['file_name', 'leftmost_pixel_x', 'leftmost_pixel_y'])
df.to_excel(output_file, index=False)



def getRightMostFromImage(mask_image):
    ret, binary_image = cv2.threshold(mask_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)

    rightmost = (-1, -1)
    if len(contours) > 0:
        cnt = contours[0]
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])  # Get rightmost point
        cv2.circle(contour_image,center,radius,(0,255,0),2)
        cv2.circle(contour_image,rightmost,5,(0,0,255),-1)

    return rightmost, contour_image

output_dir = 'C:/Users/mahsh/OneDrive/Bureau/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612/rightmosts'
output_file = "C:/Users/mahsh/OneDrive/Bureau/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612/rightmosts/rightmost_pixels.xlsx"

pipetteresults2 = []   
# Loop through each input mask image file and process it with getLeftMostFromImage function
for file in input_files:
    input_path = os.path.join(images_path, file)
    output_path = os.path.join(output_dir, file)
    # Load the input mask image
    mask_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # Process the image and get the leftmost pixel and contour image
    rightmost, ret_img = getRightMostFromImage(mask_image)
    print(f"{file}: {rightmost}")
    pipetteresults2.append([file, rightmost[0], rightmost[1]])

    # Save the resulting image with the same name as the input image to the output directory
    cv2.imwrite(output_path, ret_img)

print("Done.")

# Save results to Excel file
df = pd.DataFrame(pipetteresults2, columns=['file_name', 'rightmost_pixel_x', 'rightmost_pixel_y'])
df.to_excel(output_file, index=False)

