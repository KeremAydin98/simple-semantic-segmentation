import tensorflow as tf

class Augment(tf.keras.layers.Layer):

    def __init__(self, seed=61):

        super().__init__()

        self.augment_images = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_masks = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, images, masks):

        images = self.augment_images(images)
        masks = self.augment_masks(masks)

        return images, masks
    
