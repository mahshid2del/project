import cv2
import numpy as np
import matplotlib.pyplot as plt
from previous_script import approxPolyDP
import pandas as pd
from scipy.optimize import fsolve
from shapely.geometry import LineString, Point

# Load an image to draw the line on
cell_image = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/results/frame89_mask.png')

# Call the previous function to get the semicircle vertices
C, above, below, x1H1, y1H1, x2H1, y2H1, x1H2, y1H2, x2H2, y2H2, center, mH1, bH1, mH2, bH2 = approxPolyDP('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/frame89_mask.png')
merged_df = pd.read_excel('deformation.xlsx')

# Draw the lines on the new image
line1 = cv2.line(cell_image, (x1H1, y1H1), (x2H1, y2H1), (0, 0, 255), 3)
line2 = cv2.line(cell_image, (x1H2, y1H2), (x2H2, y2H2), (0, 0, 255), 3)

# Extract the desired rows from the DataFrame using the 'File Name' column
rows = merged_df[merged_df['file_name'] == 'frame89_mask.png']
# Extract the desired parameters from the first row of the filtered DataFrame
leftmost_pixel_x = rows.iloc[0]['leftmost_pixel_x']
leftmost_pixel_y = rows.iloc[0]['leftmost_pixel_y']
radiusperFrame = rows.iloc[0]['radiusperFrame']
deformation_perFrame = rows.iloc[0]['deformation_perFrame']
centerX = rows.iloc[0]['centerX']
centerY = rows.iloc[0]['centerY']
rightmost_pixel_x = rows.iloc[0]['rightmost_pixel_x']
rightmost_pixel_y = rows.iloc[0]['rightmost_pixel_y']
major_axis = rows.iloc[0]['major_axis']
minor_axis = rows.iloc[0]['minor_axis']

# Calculate the slope and intercept of the line
mH3 = (leftmost_pixel_y - rightmost_pixel_y) / (leftmost_pixel_x - rightmost_pixel_x)
bH3 = rightmost_pixel_y - mH3 * rightmost_pixel_x

# Extend the line to the edge of the image
height, width, _ = cell_image.shape
x1H3 = 0
y1H3 = int(bH3)
x2H3 = width - 1
y2H3 = int(mH3 * x2H3 + bH3)

# Draw the line on the new image
line3 = cv2.line(cell_image, (x1H3, y1H3), (x2H3, y2H3), (0, 0, 255), 3)
plt.imshow(cell_image)
plt.show()

# Convert the contour image to grayscale
gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
# Apply a threshold to obtain a binary image
ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
# Find the contours in the binary image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

plt.imshow(binary)
plt.show()

# Create LineString objects for each of the three lines
line1 = LineString([(x1H1, y1H1), (x2H1, y2H1)])
line2 = LineString([(x1H2, y1H2), (x2H2, y2H2)])
line3 = LineString([(x1H3, y1H3), (x2H3, y2H3)])

# Create empty lists to store the intersection points for each line
intersection_points1 = []
intersection_points2 = []
intersection_points3 = []

# Loop through the contour points and check for intersection with each line
for point in contours[0]:
    p = Point(point[0][0], point[0][1])
    if line1.contains(p):
        intersection_points1.append(p)
    if line2.contains(p):
        intersection_points2.append(p)
    if line3.contains(p):
        intersection_points3.append(p)

# Print the intersection points for each line
print("Intersection points with line 1:", intersection_points1)
print("Intersection points with line 2:", intersection_points2)
print("Intersection points with line 3:", intersection_points3)
