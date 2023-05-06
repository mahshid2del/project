

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
img = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/resized_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150)
plt.imshow(edges)
plt.show()

# Apply the Hough transform to detect circles
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Choose the circle with the closest center coordinates to the center of the image
center = (img.shape[1] // 2, img.shape[0] // 2)
selected_circle = None
min_distance = float('inf')

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        if distance < min_distance:
            selected_circle = (x, y, r)
            min_distance = distance

# Use the radius of the selected circle as the estimate of the cell radius
if selected_circle is not None:
    cell_radius = selected_circle[2]
    print("Cell radius: ", cell_radius)
else:
    print("No circle detected")
    
    
    
# import cv2
# import numpy as np

# # Load the image and convert to grayscale
# img = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/results/frame0_mask.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply edge detection
# edges = cv2.Canny(gray, 50, 150)

# # Apply the Hough transform to detect circles
# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=0, maxRadius=0)

# # Choose the circle with the closest center coordinates to the center of the image
# center = (img.shape[1] // 2, img.shape[0] // 2)
# selected_circle = None
# min_distance = float('inf')

# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")
#     for (x, y, r) in circles:
#         distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
#         if distance < min_distance:
#             selected_circle = (x, y, r)
#             min_distance = distance

# # Draw the detected circle and its radius on the input image
# if selected_circle is not None:
#     cv2.circle(img, (selected_circle[0], selected_circle[1]), selected_circle[2], (0, 255, 0), 2)
#     cv2.line(img, (selected_circle[0], selected_circle[1]), (selected_circle[0] + selected_circle[2], selected_circle[1]), (0, 255, 0), 2)
#     cv2.putText(img, str(selected_circle[2]), (selected_circle[0] + selected_circle[2] // 2, selected_circle[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Display the input image with the detected circle and its radius
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Load the image and convert to grayscale
# img = cv2.imread('C:/Users/mahsh/OneDrive/Bureau/Unet-cell/train-results/resized_image.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to segment the cell
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# # Find contours of the cell
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Find the contour with the maximum area (should be the cell)
# max_contour = max(contours, key=cv2.contourArea)

# # Fit a circle to the contour
# (x,y), (width, height), angle = cv2.minAreaRect(max_contour)
# center = (int(x),int(y))
# radius = int(min(width,height)/2)

# # Correct the aspect ratio of the detected circle
# if width > height:
#     aspect_ratio = height/width
#     radius = int(radius * (1/aspect_ratio))
# else:
#     aspect_ratio = width/height
#     radius = int(radius * (1/aspect_ratio))

# # Draw the corrected circle and its radius on the input image
# cv2.circle(img, center, radius, (0, 255, 0), 2)
# cv2.line(img, center, (center[0] + radius, center[1]), (0, 255, 0), 2)
# cv2.putText(img, str(radius), (center[0] + radius // 2, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Display the input image with the corrected circle and its radius
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
