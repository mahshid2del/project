import cv2
import numpy as np
import matplotlib.pyplot as plt
from previous_script import approxPolyDP
import pandas as pd
from scipy.optimize import fsolve

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

# Calculate the intersection point between line 1 and the contour
cnt = contours[0]
intersections1 = []
intersections2 = []
intersections3 = []

for c in contours:
    for i in range(len(cnt)):
        x, y = cnt[i][0]
        if (y > y1H1 and y < y2H1) or (y > y2H1 and y < y1H1):
            if abs(mH1*x - y + bH1) / np.sqrt(mH1**2 + 1) < 2:  # set a tolerance to handle slight deviations
                intersection_x = int((y - bH1) / mH1)
                intersection_y = y
                print('Intersection points:', intersection_x, intersection_y)

                cv2.circle(cell_image, (intersection_x, intersection_y), 5, (0, 255, 0), -1)
                intersections1.append((intersection_x, intersection_y))


# Calculate the intersection point between line 2 and the contour
for c in contours:
    for i in range(len(cnt)):
        x, y = cnt[i][0]
        if (y > y1H2 and y < y2H2) or (y > y2H2 and y < y1H2):
            if abs(mH2*x - y + bH2) / np.sqrt(mH2**2 + 1) < 2:  # set a tolerance to handle slight deviations
                intersection_x = int((y - bH2) / mH2)
                intersection_y = y
                print('Intersection points:', intersection_x, intersection_y)
                cv2.circle(cell_image, (intersection_x, intersection_y), 5, (0, 255, 0), -1)
                intersections2.append((intersection_x, intersection_y))


# Calculate the intersection point between line 3 and the contour
for c in contours:
    for i in range(len(cnt)):
        x, y = cnt[i][0]
        if (x > x1H3 and x < x2H3) or (x > x2H3 and x < x1H3):
            if abs(mH3*x - y + bH3) / np.sqrt(mH3**2 + 1) < 2:  # set a tolerance to handle slight deviations
                intersection_x = x
                intersection_y = int(mH3*x + bH3)
                print('Intersection points:', intersection_x, intersection_y)
                cv2.circle(cell_image, (intersection_x, intersection_y), 5, (0, 255, 0), -1)
                intersections3.append((intersection_x, intersection_y))

                



