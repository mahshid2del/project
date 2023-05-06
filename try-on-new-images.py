import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.utils import CustomObjectScope
from train import iou
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt 


# Load the trained U-Net model
with CustomObjectScope({'iou': iou}):
    model = tf.keras.models.load_model('C:/Users/mahsh/OneDrive/Bureau/inner_circle/files/model.h5')

# Define the size of the input images to the model
input_shape = (256, 256, 3)

# Define the path to the directory containing the new polyp images
image_dir = 'C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/alldata'


# Load and preprocess the images
images = []
image_paths = sorted(os.listdir(image_dir))  # Sort the list of image paths
for image_path in image_paths:
    image = cv2.imread(os.path.join(image_dir, image_path))
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    images.append(image)
images = np.array(images)

random_indices = np.random.choice(len(images), 10, replace=False)
random_images = images[random_indices]

# Display the randomly selected images
plt.figure(figsize=(20, 10))
columns = 5
for i, image in enumerate(random_images):
    plt.subplot(2, columns, i + 1)
    plt.imshow(image)
plt.show()

# Feed the preprocessed images through the U-Net model to obtain the segmentation masks
masks = model.predict(images)


random_indices = np.random.choice(len(masks), 10, replace=False)
random_masks = masks[random_indices]

# Display the randomly selected images
plt.figure(figsize=(20, 10))
columns = 5
for i, mask in enumerate(random_masks):
    plt.subplot(2, columns, i + 1)
    plt.imshow(mask)
plt.show()


# # Save the final segmentation masks to disk
# save_dir = 'C:/Users/mahsh/OneDrive/Bureau/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/CVC-612/results'

# # Save each predicted mask to a separate file
# for i, mask in enumerate(masks):
#     mask = mask.squeeze() * 255.0
#     mask = mask.astype(np.uint8)
#     cv2.imwrite(os.path.join(save_dir, f'{str(i+1)}.png'), mask)

# Save the final segmentation masks to disk
save_dir = 'C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results/results'

# Save each predicted mask to a separate file with the same name as the input image
for i, mask in enumerate(masks):
    image_path = image_paths[i]  # Get the corresponding image file name
    mask = cv2.resize(mask, (1280, 720))
    mask = mask.squeeze() * 255.0
    mask = mask.astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{image_path[:-4]}_mask.png'), mask)

##########################
