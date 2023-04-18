import tensorflow as tf
import numpy as np
import config
import os


def preprocess(image):

  # Resize the input image and mask using TensorFlow operations
  input_image = tf.image.resize(image[:, :256, :], (128, 128))
  input_mask = tf.image.resize(image[:, 256:, :], (128, 128))

  # Normalize the input image and cast the input mask to integer type using TensorFlow operations
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.int32)


  return input_image, input_mask


def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channel=3)

    input_image, input_mask =  preprocess(image)

    mask = np.zeros(shape=(128, 128), dtype = np.uint32)

    for row in range(input_mask.shape[0]):
        for col in range(input_mask.shape[1]):
            a = input_mask[row, col, :]

            # Compute the distance between the pixel color and each label color
            distances = np.sqrt(np.sum(np.square(a - np.array(list(config.id_map.values()))), axis=1))

            # Find the index of the label with the smallest distance
            mask[row, col] = np.argmin(distances)


    return input_image, mask

def load_images(image_directory):

    images_dir = os.listdir(image_directory)

    image_paths = [image_directory + image_path for image_path in images_dir]

    images = []
    masks = []

    for image_path in image_paths:

        input_image, input_mask = load_image(image_path)

        images.append(input_image)
        masks.append(input_mask)

    return images, masks





