import cv2
import numpy as np
import matplotlib.pyplot as plt
from previous_script import approxPolyDP
from tip_micropipette import getLeftMostFromImage


# Load an image to draw the line on
cell_image = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/bb/frame89_mask.png')
image = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/frame89_mask.png', 0)
# Call the previous function to get the semicircle vertices
C, above, below, x1H1, y1H1, x2H1, y2H1 , x1H2, y1H2, x2H2, y2H2 = approxPolyDP(image)
leftmost, contour_image = getLeftMostFromImage(image)
# Draw the line on the new image
cv2.line(cell_image, (x1H1, y1H1), (x2H1, y2H1), (0, 0, 255), 3)

cv2.line(cell_image, (x1H2, y1H2), (x2H2, y2H2), (0, 0, 255), 3)


# Display the result
plt.imshow(cell_image)
plt.show()


micropipette_image = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/bb/frame89_mask.png')
