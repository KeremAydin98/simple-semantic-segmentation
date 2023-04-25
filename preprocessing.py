import tensorflow as tf
import numpy as np
import config
import os

def preprocess(image, mask, label_dict):

    # Normalize the input image and cast the input mask to integer type using TensorFlow operations
    input_image = tf.cast(image, tf.float32) / 255.0
    input_mask = tf.cast(mask, tf.int32)
    
    mask = np.zeros(shape=(720, 960))
    
    for x in range(720):
        for y in range(960):

            mask[x,y] = label_dict[str(np.array(input_mask[x,y,:]))]

    return input_image, mask


def load_image(image_path, mask_path, label_dict):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask)

    image, mask = preprocess(image, mask, label_dict)

    return image, mask


def load_images(image_directory, label_dict):

    images_dir = os.listdir(image_directory)
    masks_dir = os.listdir(image_directory + "_labels/")

    image_paths = [image_directory + "/" + image_path for image_path in images_dir]
    masks_paths = [image_directory + "_labels" + "/" + mask_path for mask_path in masks_dir]

    images = []
    masks = []

    for image_path, mask_path in zip(image_paths,masks_paths):

        input_image, input_mask = load_image(image_path, mask_path, label_dict)

        images.append(input_image)
        masks.append(input_mask)

    return images, masks





