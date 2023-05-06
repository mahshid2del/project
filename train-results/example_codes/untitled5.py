import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt


def read_json_file(filename):
    """
    Reads in a JSON file containing labeled regions data and returns a dictionary.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def read_image(filename):
    """
    Reads in an image file and returns a numpy array.
    """
    img = np.asarray(PIL.Image.open(filename))
    return img


def apply_labels_to_image(img, regions):
    """
    Applies labeled regions to an image and returns a new image with the regions drawn on it.
    """
    if regions != {}:
        try:
            shape_x = regions['0']['shape_attributes']['all_points_x']
            shape_y = regions['0']['shape_attributes']['all_points_y']
        except KeyError:
            shape_x = regions[0]['shape_attributes']['all_points_x']
            shape_y = regions[0]['shape_attributes']['all_points_y']
        ab = np.stack((shape_x, shape_y), axis=1)
        img = cv2.drawContours(img, [ab], -1, (255, 255, 255), -1)
    return img


def create_mask_from_labels(img, regions):
    """
    Creates a binary mask from labeled regions in an image and returns the mask as a numpy array.
    """
    mask = np.zeros((img.shape[0], img.shape[1]))
    if regions != {}:
        try:
            shape_x = regions['0']['shape_attributes']['all_points_x']
            shape_y = regions['0']['shape_attributes']['all_points_y']
        except KeyError:
            shape_x = regions[0]['shape_attributes']['all_points_x']
            shape_y = regions[0]['shape_attributes']['all_points_y']
        ab = np.stack((shape_x, shape_y), axis=1)
        mask = cv2.drawContours(mask, [ab], -1, 255, -1)
    return mask


def save_image(filename, img):
    """
    Saves a numpy array as an image file.
    """
    cv2.imwrite(filename, img.astype(np.uint8))


def process_images(data, image_dir, label_dir):
    """
    Processes images and labels from a dictionary of labeled regions data and saves labeled images as separate files.
    """
    filenames = os.listdir(image_dir)
    for i, (filename, regions) in enumerate(data.items()):
        if filename in filenames:
            img = read_image(os.path.join(image_dir, filename))
            img = apply_labels_to_image(img, regions['regions'])
            save_image(os.path.join(label_dir, f"{i:05d}.png"), create_mask_from_labels(img, regions['regions']))


if __name__ == '__main__':
    json_file = "via_project_4Apr2023_12h20m_coco.json"
    image_dir = "image"
    label_dir = "label"
    data = read_json_file(json_file)
    process_images(data, image_dir, label_dir)
