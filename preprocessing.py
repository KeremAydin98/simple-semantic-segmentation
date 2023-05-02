import tensorflow as tf
import numpy as np
import os
import config

def preprocess(image):

  # Resize the input image and mask using TensorFlow operations
  input_image = image[:, :256, :]
  input_mask = image[:, 256:, :]

  input_image = tf.image.central_crop(input_image, 0.5)
  input_mask = tf.image.central_crop(input_mask, 0.5) 

  # Normalize the input image and cast the input mask to integer type using TensorFlow operations
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.int32)


  return input_image, input_mask


def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)

    input_image, input_mask =  preprocess(image)

    # Precompute the distances between all pairs of pixel colors and label colors
    label_colors = np.array(list(config.id_map.values()))
    distances = np.sqrt(np.sum(np.square(input_mask[:, :, np.newaxis, :] - label_colors[np.newaxis, np.newaxis, :, :]), axis=3))

    # Find the index of the label with the smallest distance for each pixel
    mask = np.argmin(distances, axis=2)

    return input_image, mask

def load_images(image_directory, n_dataset=None):

    images_dir = os.listdir(image_directory)

    image_paths = [image_directory + image_path for image_path in images_dir]

    images = []
    masks = []

    for i, image_path in enumerate(image_paths):

        input_image, input_mask = load_image(image_path)

        images.append(input_image)
        masks.append(input_mask)

        if n_dataset:
          print(f"Done {i+1}/{n_dataset}")
          if i+1 == n_dataset:
            break
        else:
          print(f"Done {i+1}/{len(image_paths)}")

    return images, masks