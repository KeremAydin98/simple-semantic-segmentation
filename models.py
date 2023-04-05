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
    
class U_net:

    def __init__(self):
        super().__init__()

        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        layer_names = [
            "block_1_expand_relu", # 64x64
            "block_3_expand_relu", # 32x32
            "block_6_expand_relu", # 16x16
            "block_13_expand_relu", # 8x8
            "block_16_project", # 4x4
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        self.down_stack = tf.keras.models.Model(inputs=base_model.input, outputs=base_model_outputs)
        self.down_stack.trainable = False

    def upsample(self, filters, size):

        """
        Upsamples an input.

        Conv2DTranspose => Batchnorm => Relu

        """

        result = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        return result
    
    def create_upstack(self):

        upsample_1 = self.upsample(512, 3)
        upsample_2 = self.upsample(256, 3)
        upsample_3 = self.upsample(128, 3)
        upsample_4 = self.upsample(64, 3)

        upstack = [upsample_1,
                    upsample_2,
                    upsample_3,
                    upsample_4]

        return upstack
    
    def create_model(self, output_channels):

        upstack = self.create_upstack()

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])

        # Downsampling
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
    
        # Upsampling and establishing skip connections
        for up, skip in zip(upstack, skips):

            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer
        last = tf.keras.layers.Conv2DTranspose(
            filters = output_channels, kernel_size=3, strides=2, padding='same'
        )

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
