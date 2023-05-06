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
# Define the parameters for the ellipse
center = (centerX, centerY)
axes = (major_axis, minor_axis)
angle = 90  # Angle of rotation of the ellipse
startAngle = 0  # Starting angle of the elliptical arc in degrees
endAngle = 360  # Ending angle of the elliptical arc in degrees
color = (255, 0, 0)  # Color of the ellipse
thickness = 2  # Thickness of the ellipse boundary

# Draw the ellipse on the image
ellipse = cv2.ellipse(cell_image, (int(centerX), int(centerY)), (int(major_axis/2), int(minor_axis/2)), angle, startAngle, endAngle, color, thickness)

# Create LineString objects for each of the three lines
line1 = LineString([(x1H1, y1H1), (x2H1, y2H1)])
line2 = LineString([(x1H2, y1H2), (x2H2, y2H2)])
line3 = LineString([(x1H3, y1H3), (x2H3, y2H3)])

# Create empty lists to store the intersection points for each line
intersection_points1 = []
intersection_points2 = []
intersection_points3 = []
# Loop through the ellipse points and check for intersection with each line
for i in range(startAngle, endAngle + 1, 10):
    # Calculate the x and y coordinates of the point on the ellipse
    x = int(centerX + (major_axis/2) * np.cos(np.radians(i)))
    y = int(centerY - (minor_axis/2) * np.sin(np.radians(i)))
    point = Point(x, y)

    # Check for intersection with line 1
    if line1.intersects(point):
        intersection_points1.append(point)
    
    # Check for intersection with line 2
    if line2.intersects(point):
        intersection_points2.append(point)

    # Check for intersection with line 3
    if line3.intersects(point):
        intersection_points3.append(point)

# Print the intersection points for each line
print("Intersection points with line 1:", intersection_points1)
print("Intersection points with line 2:", intersection_points2)
print("Intersection points with line 3:", intersection_points3)

