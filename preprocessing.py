import tensorflow as tf

def normalize(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask

def load_image(dataset):

    input_image = tf.image.resize(dataset['image'], (128, 128))
    input_mask = tf.image.resize(dataset['segmentation_mask'], (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

