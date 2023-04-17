import tensorflow as tf
import numpy as np
import os


def preprocess(image):

    input_image, input_mask =  np.array(image)[:,:256, :], np.array(image)[:,256:, :]

    input_image = tf.image.resize(input_image, (128, 128))
    input_mask = tf.image.resize(input_mask, (128, 128))

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.int32)

    return input_image, input_mask



def load_image(image_path):

    image = tf.keras.utils.load_img(image_path)

    input_image, input_mask =  preprocess(image)

    mask = np.zeros(shape=(128, 128), dtype = np.uint32)

    for row in range(128):
        for col in range(128):
            a = input_mask[row, col, :]
            final_key = None
            final_d = None
            for key, value in id_map.items():
                d = np.sum(np.sqrt(pow(a - value, 2)))
                if final_key == None:
                    final_d = d
                    final_key = key
                elif d < final_d:
                    final_d = d
                    final_key = key
            mask[row, col] = final_key

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

