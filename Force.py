import math
import cv2
import glob
import natsort
from new_script import calculate_radius

# Define conversion factor from pixels to micrometers
pixel_to_um = 0.1  



# Define a list of image paths
image_paths = glob.glob('C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results/*.png')
image_paths = natsort.natsorted(image_paths)

# Calculate epsilon_d, sigma_d, and F for each image in the list
for image_path in image_paths:
    # Load the image and calculate the radius
    cell_image = cv2.imread(image_path)
    deformation_perFrame, r = calculate_radius(image_path)
    
    # Calculate the variables for the equation
    a = r / pixel_to_um
    wd = deformation_perFrame / pixel_to_um
    # Define the variables used in the equation
    c = 0.5 * a  # micropipette radius
    E = 1.0  # elastic modulus
    v = 0.3  # Poisson's ratio
    h = 0.01  # membrane thickness
    
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

    # Print the results
    print(f"Image: {image_path}, radius: {r:.2f} pixels, deformation: {deformation_perFrame:.2f} pixels, force: {F:.2f} N")

