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

##################################################

# #deformation : 
# deformation_results = []
# for cell_result, pipette_result in zip(cellLocation, pipetteresults):
#     # extract relevant data
#     file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y = cell_result
#     file_name, leftmost_x, leftmost_y = pipette_result
    
#     # perform deformation calculation
#     radius = Minor_axis/2 
#     print(f"radius for {file_name}: {radius:.2f}")
#     distance = ((leftmost_x - center_x)**2 + (leftmost_y - center_y)**2)**0.5
#     print(f"distance for {file_name}: {distance:.2f}")

#     deformation = distance - radius
#     # print result
#     print(f"Deformation for {file_name}: {deformation:.2f}")


#     # print result based on deformation value
#     if deformation > 2:
#         deformation_results.append((file_name, deformation, "no deformation"))
#         print("there is no deformation : ", file_name, deformation)
#     elif deformation < 0:
#         deformation_results.append((file_name, deformation, "deformation"))
#         print("there is deformation : ", file_name, deformation)
#     elif 0 <= deformation <= 2:
#         deformation_results.append((file_name, deformation, "contact"))
#         print("there is a contact : ", file_name, deformation)

# # sort the results based on file_name
# sorted_results = sorted(deformation_results, key=lambda x: int(x[0].split('_')[0][5:]))

# # print the sorted results
# for result in sorted_results:
#     print(f"{result[0]}: {result[1]:.2f} - {result[2]}")


#################################################################################



# def calculate_deformation(cellLocation, pipetteresults):
#     deformation_results = []
#     for cell_result, pipette_result in zip(cellLocation, pipetteresults):
#         # extract relevant data
#         file_name, Major_axis, Minor_axis, Width, Height, center_x, center_y = cell_result
#         file_name, leftmost_x, leftmost_y = pipette_result

#         # perform deformation calculation
#         radius = Minor_axis/2 
#         distance = ((leftmost_x - center_x)**2 + (leftmost_y - center_y)**2)**0.5
#         deformation = distance - radius

#         # add result to deformation_results list
#         if deformation > 2:
#             deformation_results.append((file_name, deformation, "no deformation"))
#         elif deformation < 0:
#             deformation_results.append((file_name, deformation, "deformation"))
#         elif 0 <= deformation <= 2:
#             deformation_results.append((file_name, deformation, "contact"))

#     # sort the results based on file_name
#     sorted_results = sorted(deformation_results, key=lambda x: int(x[0].split('_')[0][5:]))

#     # return sorted results
#     return sorted_results

# results = calculate_deformation(cellLocation, pipetteresults)
# for result in results:
#     print(f"{result[0]}: {result[1]:.2f} - {result[2]}")

###############################################################################
def calculate_deformation(cellLocation, pipetteresults):
    deformation_results = []
    
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
        deformation = distance - radius

        # add result to deformation_results list
        if deformation < 0:
            deformation_results.append((file_name, deformation, "deformation"))

    # sort the results based on file_name
    sorted_results = sorted(deformation_results, key=lambda x: int(x[0].split('_')[0][5:]))

    # return sorted results
    return sorted_results


results = calculate_deformation(cellLocation, pipetteresults)
for result in results:
    print(f"{result[0]}: {result[1]:.2f} - {result[2]}")
 
    
###############################################################################
#approxPolyDP
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


# Define a function to calculate the dimple radius for a given image
def calculate_radius(image_path, deformation_results):
    # Load an image
    cell_image = cv2.imread(image_path)

    # Call the previous function to get the semicircle vertices
    C, above, below, c_above_distance, c_below_distance = approxPolyDP(image_path)

    # Initialize the radius as None
    r = None

    # Loop over the deformation results and find the corresponding image file
    for deformation_result in deformation_results:
        file_name, deformation, deformation_type = deformation_result
        if file_name in image_path and deformation_type == 'deformation':
            # Calculate the dimple radius
            if c_above_distance is not None and c_below_distance is not None:
                r = sqrt((deformation/2)**2 + ((float(c_above_distance) + float(c_below_distance))/2)**2)
            else:
                r = 0
            break

    return r, deformation



# Define a list of image paths
image_paths = glob.glob('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/*.png')
image_paths = natsort.natsorted(image_paths)

# Calculate the radius for each image in the list
radii = []
for image_path in image_paths:
    r, deformation = calculate_radius(image_path, results)
    radii.append(r)
    print(f"Image: {image_path}, radius: {str(r)}")


####################################################################################
#force 
# Create an empty dataframe to store the results
results_df = pd.DataFrame(columns=['File Name', 'Radius', 'Deformation', 'Epsilon_d', 'Sigma_d', 'Force'])

# Calculate the force for each image in the list
for image_path in image_paths:
    r, deformation = calculate_radius(image_path, results)
    if r is not None:
        pixel_to_um = 0.1 
        wd = deformation / pixel_to_um
        c = 0.5  # micropipette radius
        E = 1.0  # elastic modulus
        v = 0.3  # Poisson's ratio
        h = 0.01  # membrane thickness
        
        a = r / pixel_to_um
        t = (2 * wd) / (a - c)
    
        # Calculate the deformation of the membrane
        term1 = ((a * math.sqrt(1 + t ** 2)) - 2 * a) / (2 * (a + c))
        term2 = (a * math.log(t + math.sqrt(1 + t ** 2)) + 2 * wd) / (2 * t * (a + c))
        term3 = (2 * wd - 2 * wd * (1 - t ** 2) ** 2 / 3) / (3 * t ** 3 * (a + c))
        epsilon_d = term1 + term2 + term3
    
        # Calculate the stress in the membrane
        sigma_d = E / (1 - v) * epsilon_d
    
        # Calculate the force on the membrane
        term4 = 1 - (c / a) ** 2 + math.log(c / a) ** 2
        F = - (4 * math.pi * wd * sigma_d * h) / term4
        
        print(f"Image: {image_path}, force: {F}")
        
        # Update the results dataframe with the force value
        results_df.loc[results_df['File Name'] == os.path.basename(image_path), 'Force'] = F
        results_df.loc[results_df['File Name'] == os.path.basename(image_path), 'Radius'] = r
        results_df.loc[results_df['File Name'] == os.path.basename(image_path), 'Deformation'] = deformation
        results_df.loc[results_df['File Name'] == os.path.basename(image_path), 'Epsilon_d'] = epsilon_d
        results_df.loc[results_df['File Name'] == os.path.basename(image_path), 'Sigma_d'] = sigma_d
    else:
        a = 0
        r = 0
        wd = 0
        deformation =0
        epsilon_d = 0
        sigma_d = 0
        F = 0
        print(f"Image: {image_path}, radius not found")
    results_df = pd.concat([results_df, pd.DataFrame({
        
        'File Name': [os.path.basename(image_path)],
        'Radius': [r],
        'Deformation': [deformation],
        'Epsilon_d': [epsilon_d],
        'Sigma_d': [sigma_d],
        'Force': [F]}, index=[0])])

# Save the dataframe to an excel file
results_df.to_excel('C:/Users/mahsh/OneDrive/Bureau/inner_circle/Force.xlsx', index=False)


import matplotlib.pyplot as plt

# Create a list of numbers from 0 to 462
x_values = list(range(0, 463))

# Plot deformation vs file name
plt.figure(figsize=(10,5))
plt.plot(x_values, abs(results_df['Deformation']), 'bo-')
plt.xlabel('File Index')
plt.ylabel('Deformation (pixels)')
plt.title('Deformation vs File Index')
#plt.xticks(rotation=90)
plt.show()

# Plot radius vs index
plt.figure(figsize=(10,5))
plt.plot(x_values, results_df['Radius'], 'ro-')
plt.xlabel('Index')
plt.ylabel('Radius (pixels)')
plt.title('Radius vs Index')
plt.show()

# Plot force vs index
plt.figure(figsize=(10,5))
plt.plot(x_values, results_df['Force'], 'go-')
plt.xlabel('Index')
plt.ylabel('Force (N/m^2)')
plt.title('Force vs Index')
plt.show()

#######################################################
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Plot deformation vs index
def animate(i):
    plt.cla() # clear previous plot
    plt.plot(x_values[:i], abs(results_df['Deformation'][:i]), 'bo-')
    plt.xlabel('Frame')
    plt.ylabel('Deformation (pixels)')
    plt.title('Deformation vs Frame')
    plt.tight_layout()

# Create the animation
fig, ax = plt.subplots(figsize=(10,5))
ani = animation.FuncAnimation(fig, animate, frames=len(results_df), interval=50)

# Save the animation as a GIF using PillowWriter
writer = PillowWriter(fps=20)
ani.save('C:/Users/mahsh/OneDrive/Bureau/inner_circle/deformation.gif', writer=writer)

# Show the plot
plt.show()

# Plot radius vs index
def animate(i):
    plt.cla() # clear previous plot
    plt.plot(x_values[:i], results_df['Radius'][:i], 'ro-')
    plt.xlabel('Frame')
    plt.ylabel('Radius (pixels)')
    plt.title('Radius vs Frame')
    plt.tight_layout()

# Create the animation
fig, ax = plt.subplots(figsize=(10,5))
ani = animation.FuncAnimation(fig, animate, frames=len(results_df), interval=50)

# Save the animation as a GIF using PillowWriter
writer = PillowWriter(fps=20)
ani.save('C:/Users/mahsh/OneDrive/Bureau/inner_circle/radius.gif', writer=writer)

# Show the plot
plt.show()

# Plot force vs index
def animate(i):
    plt.cla() # clear previous plot
    plt.plot(x_values[:i], results_df['Force'][:i], 'go-')
    plt.xlabel('Frame')
    plt.ylabel('Force (N/m^2)')
    plt.title('Force vs Frame')
    plt.tight_layout()

# Create the animation
fig, ax = plt.subplots(figsize=(10,5))
ani = animation.FuncAnimation(fig, animate, frames=len(results_df), interval=50)

# Save the animation as a GIF using PillowWriter
writer = PillowWriter(fps=20)
ani.save('C:/Users/mahsh/OneDrive/Bureau/inner_circle/force.gif', writer=writer)

# Show the plot
plt.show()

