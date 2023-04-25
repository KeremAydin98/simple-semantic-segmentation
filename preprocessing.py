import tensorflow as tf
import numpy as np
import config
import os

def preprocess(image, mask, label_dict):

    # Normalize the input image and cast the input mask to integer type using TensorFlow operations
    input_image = tf.cast(image, tf.float32) / 255.0
    input_mask = tf.cast(mask, tf.int32)

    input_image = tf.image.resize(input_image, (128, 128))
    input_mask = tf.image.resize(input_mask, (128, 128))
    
    mask = np.zeros(shape=(128, 128))
    
    for i, label in label_dict.items():
        mask[np.all(input_mask == np.array(i).astype(int), axis=-1)] = label

    return input_image, mask


def load_image(image_path, mask_path, label_dict):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask)

    mask = tf.reduce_sum(mask * tf.constant([1, 256, 256*256], dtype=tf.int32), axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = tf.map_fn(lambda x: label_dict[str(x.numpy())], mask, dtype=tf.int32)

    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    
    mask = tf.image.resize(mask[..., tf.newaxis], (128, 128))

    return image, mask[..., 0]



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





