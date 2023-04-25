import tensorflow as tf

# Data augmentation class
class Augment(tf.keras.layers.Layer):

    def __init__(self, seed=61):

        super().__init__()

        self.augment_images = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_masks = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, images, masks):

        images = self.augment_images(images)
        masks = self.augment_masks(masks)

        return images, masks
    

# U-net model structure class
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

        # Entire left hand of U-net does not need weight adjustments
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
    
    def call(self, inputs):

        upstack = self.create_upstack()

        # Downsampling
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
    
        # Each stage of the decoder upstack is concatenated with the corresponding layer of the encoder down stack
        # Upsampling and establishing skip connections
        for up, skip in zip(upstack, skips):

            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer
        last = tf.keras.layers.Conv2DTranspose(
            filters = 1, kernel_size=3, strides=2, padding='same'
        )

        outputs = last(x)

        return outputs


# FCN model structure class
class FCN(tf.keras.Model):

  def __init__(self):

    # Feature extractor
    self.base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')


    self.f4_conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same',
                      activation=None)

    # Covolutions instead of dense layers
    self.f5 = tf.keras.Sequential([
                      tf.keras.layers.Conv2D(filters=4086, kernel_size=7, padding='same',activation='relu'),
                      tf.keras.layers.Dropout(0.5),
                      tf.keras.layers.Conv2D(filters=4086, kernel_size=7, padding='same',activation='relu'),
                      tf.keras.layers.Dropout(0.5),
                      tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same', activation=None)])


    # Using a transposed conv (w/ s=2) to upscale `f5` into a 14 x 14 map
    # so it can be merged with features from `f4_conv1` obtained from `f4`
    self.f5_conv3_upsample = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2,
                                use_bias=False, padding='same', activation='relu')
    
    # We repeat the operation to merge `merge1` and `f3` into a 28 x 28 map:
    self.f3_conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same',
                      activation=None)
    
    self.merge_upsample = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2,
                                use_bias=False, padding='same', activation='relu')
    
    # Final layer
    self.final = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=16, strides=8,
                              padding='same', activation=None)
    
  def call(self, inputs):

    x = self.base_model(inputs)

    f3 = x.get_layer('block3_pool').output  
    f4 = x.get_layer('block4_pool').output  
    f5 = x.get_layer('block5_pool').output  


    f5_out = self.f5(f5)

    upsampled_f5 = self.f5_conv3_upsample(f5_out)
    f4_out  = self.f4_conv1(f4)

    # Merging the 2 feature maps (addition):
    merge1 = tf.keras.layers.add([upsampled_f5, f4_out]) 

    merge1_x2 = self.merge_upsample(merge1)

    f3_out = self.f3_conv1(f3)

    merge2 = tf.keras.layers.add([f3_out, merge1_x2])

    outputs = self.final(merge2)

    return outputs


class DeepLabV3:

    def __init__(self):

        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        layer_names = [
            "block_1_expand_relu", # 64x64
            "block_3_expand_relu", # 32x32
            "block_6_expand_relu", # 16x16
            "block_13_expand_relu", # 8x8
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Entire left hand of U-net does not need weight adjustments
        self.down_stack = tf.keras.models.Model(inputs=base_model.input, outputs=base_model_outputs)
        self.down_stack.trainable = False

        self.upsampling = tf.keras.layers.UpSampling2D(interpolation="bilinear")

        self.ConvUpscaleBlock = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, stride=[2, 2], activation="relu")
        ])

        self.ConvBlock = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, activation="relu")
        ])

    def AtrousSpatialPyramidPoolingModule(self, inputs):

        image_features = tf.reduce_mean(inputs, [1,2], keep_dims=True)

        image_features = tf.keras.layers.Conv2D(256, 1, activation="relu")(image_features)
        image_features = tf.keras.layers.Upsampling2D(interploation="bilinear")(image_features)

        atrous_block_1 = tf.keras.layers.Conv2D(256, 1, activation="relu")(image_features)
        atrous_block_6 = tf.keras.layers.Conv2D(256, 3, dilation_rate=6, activation="relu")(image_features)
        atrous_block_12 = tf.keras.layers.Conv2D(256, 3, dilation_rate=12, activation="relu")(image_features)
        atrous_block_18 = tf.keras.layers.Conv2D(256, 3, dilation_rate=18, activation="relu")(image_features)

        net = tf.concat((image_features, atrous_block_1, atrous_block_6, atrous_block_12, atrous_block_18), axis=3)
        net = tf.keras.layers.Conv2D(256, 1, activation="relu")(net)

        return net 

    def call(self, inputs):

        x = self.down_stack(inputs)

        x = self.AtrousSpatialPyramidPoolingModule(x)

        x = self.upsampling(x)

        outputs = tf.keras.layers.Conv2D(256, 31, [1,1], activation="softmax")(x)

        return outputs
    

class DeepLabV3_plus:

    def __init__(self):

        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        layer_names = [
            "block_1_expand_relu", # 64x64
            "block_3_expand_relu", # 32x32
            "block_6_expand_relu", # 16x16
            "block_13_expand_relu", # 8x8
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Entire left hand of U-net does not need weight adjustments
        self.down_stack = tf.keras.models.Model(inputs=base_model.input, outputs=base_model_outputs)
        self.down_stack.trainable = False

        self.ConvUpscaleBlock = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, stride=[2, 2], activation="relu")
        ])

        self.ConvBlock = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, 3, activation="relu")
        ])

    def AtrousSpatialPyramidPoolingModule(self, inputs):

        image_features = tf.reduce_mean(inputs, [1,2], keep_dims=True)

        image_features = tf.keras.layers.Conv2D(256, 1, activation="relu")(image_features)
        image_features = tf.keras.layers.Upsampling2D(interploation="bilinear")(image_features)

        atrous_block_1 = tf.keras.layers.Conv2D(256, 1, activation="relu")(image_features)
        atrous_block_6 = tf.keras.layers.Conv2D(256, 3, dilation_rate=6, activation="relu")(image_features)
        atrous_block_12 = tf.keras.layers.Conv2D(256, 3, dilation_rate=12, activation="relu")(image_features)
        atrous_block_18 = tf.keras.layers.Conv2D(256, 3, dilation_rate=18, activation="relu")(image_features)

        net = tf.concat((image_features, atrous_block_1, atrous_block_6, atrous_block_12, atrous_block_18), axis=3)
        net = tf.keras.layers.Conv2D(256, 1, activation="relu")(net)

        return net 

    def call(self, inputs):

        encoder_features = self.down_stack(inputs)

        x = self.AtrousSpatialPyramidPoolingModule(encoder_features)
        x = tf.keras.layers.Conv2D(256, 1, activation="relu")(x)
        decoder_features = self.upsampling(x)

        encoder_features = tf.keras.layers.Conv2D(48, 1, activation="relu")(encoder_features)

        net = tf.concat((encoder_features, decoder_features), axis=3)

        net = tf.keras.layers.Conv2D(256, 3, activation="relu")(net)
        net = tf.keras.layers.Conv2D(256, 3, activation="relu")(net)
        
        net = tf.keras.layers.UpSampling2D(interpolation="bilinear")(net)

        outputs = tf.keras.layers.Conv2D(256, 31, [1,1], activation="softmax")(net)

        return outputs



