import tensorflow as tf
import numpy as np

def normalize(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = input_mask / np.max(input_mask)

    return input_image, input_mask

def load_image(image_path, mask_path):

    image = tf.keras.utils.load_img(image_path)
    mask = tf.keras.utils.load_img(mask_path)

    input_image = tf.image.resize(image, (128, 128))
    input_mask = tf.image.resize(mask, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_images(image_paths, mask_paths):

    images = []
    masks = []

    for image_path, mask_path in zip(image_paths, mask_paths):

        image, mask = load_image(image_path, mask_path)

        images.append(image)
        masks.append(mask)

    return images, masks

