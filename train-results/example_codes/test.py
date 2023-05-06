import cv2
import numpy as np
import glob
import os
import csv

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
images_path = 'C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/results'

# Get a list of all image files in the directory
input_files = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
output_dir = 'C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/leftmosts'

# Create a new CSV file to write the leftmost pixels and file names to
csv_file_path = os.path.join(output_dir, 'leftmost_pixels.csv')
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['file_name', 'leftmost_pixel_x', 'leftmost_pixel_y'])
    
    # Loop through each input mask image file and process it with getLeftMostFromImage function
    for file in input_files:
        input_path = os.path.join(images_path, file)
        output_path = os.path.join(output_dir, file)
        # Load the input mask image
        mask_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        # Process the image and get the leftmost pixel and contour image
        leftmost, ret_img = getLeftMostFromImage(mask_image)
        print(f"{file}: {leftmost}")
        # Save the resulting image with the same name as the input image to the output directory
        cv2.imwrite(output_path, ret_img)
        # Write the file name and leftmost pixel to the CSV file
        writer.writerow([file, leftmost[0], leftmost[1]])
print("Done.")



import cv2
import numpy as np

img = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/results/frame0_mask.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)