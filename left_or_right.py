# import cv2

# # Load the binary mask
# img = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612/results/frame0_mask.png', cv2.IMREAD_GRAYSCALE)

# # apply thresholding to convert to binary
# _, img = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# # show the binary image
# cv2.imshow('Binary Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Find the first white pixel from the left and the right side of the image
# for i in range(img.shape[0]):
#     if img[i, 0] == 255:
#         print("pipette enters from the left side.")
#         break

#     if img[i, -1] == 255:
#         print("pipette enters from the right side.")
#         break

import cv2
import numpy as np

# Load image
image = cv2.imread("C:/Users/mahsh/OneDrive/Bureau/frame423_mask.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create binary mask
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding box of largest contour
largest_contour = max(contours, key=cv2.contourArea)
x,y,w,h = cv2.boundingRect(largest_contour)

# Check if object enters from right side
if x + w == image.shape[1]:
    print("Object enters from the right side")
else:
    print("Object does not enter from the right side")

# Rotate image if necessary
if x + w == image.shape[1]:
    angle = np.arctan2(y+h/2 - image.shape[0]/2, x+w/2 - image.shape[1]/2) * 180/np.pi
    rotated = cv2.rotate(image, cv2.ROTATE_180)

# Save rotated image
cv2.imwrite("C:/Users/mahsh/OneDrive/Bureau/rotated_image.png", rotated)
