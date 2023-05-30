import tensorflow as tf

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

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)
    
class U_net(tf.keras.Model):

    def __init__(self):
        super().__init__()

        base_model = tf.keras.applications.MobileNetV2(input_shape=[224,224, 3], weights="imagenet", include_top=False)

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

        
        # Upstack layers
        self.upsample_1 = tf.keras.Sequential([
                          tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same', use_bias=False),
                          tf.keras.layers.BatchNormalization(),
                          tf.keras.layers.ReLU()
                      ])
        self.upsample_2 = tf.keras.Sequential([
                          tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', use_bias=False),
                          tf.keras.layers.BatchNormalization(),
                          tf.keras.layers.ReLU()
                      ])
        self.upsample_3 = tf.keras.Sequential([
                          tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False),
                          tf.keras.layers.BatchNormalization(),
                          tf.keras.layers.ReLU()
                      ])
        self.upsample_4 = tf.keras.Sequential([
                          tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', use_bias=False),
                          tf.keras.layers.BatchNormalization(),
                          tf.keras.layers.ReLU()
                      ])
        
        self.upstack = [self.upsample_1, self.upsample_2, self.upsample_3, self.upsample_4]

        # This is the last layer
        self.last_layer = tf.keras.layers.Conv2DTranspose(
            filters = 31, kernel_size=3, strides=2, padding='same', activation="softmax"
        )
    
    def call(self, inputs):

        # Downsampling
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
    
    
        # Each stage of the decoder upstack is concatenated with the corresponding layer of the encoder down stack
        # Upsampling and establishing skip connections
        for up, skip in zip(self.upstack, skips):

            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        # This is the last layer
        outputs = self.last_layer(x)

        return outputs



    
# FCN model structure class
class FCN(tf.keras.Model):

  def __init__(self):

    super().__init__()

    # Feature extractor
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=[128, 128, 3], weights='imagenet')

    layer_names = [
            "block3_pool",
            "block4_pool",
            "block5_pool", 
        ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Entire left hand of U-net does not need weight adjustments
    self.down_stack = tf.keras.models.Model(inputs=base_model.input, outputs=base_model_outputs)

    self.f4_conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same',
                      activation='relu')

    # Covolutions instead of dense layers
    self.f5 = tf.keras.Sequential([
                      tf.keras.layers.Conv2D(filters=512, kernel_size=7, padding='same',activation='relu'),
                      tf.keras.layers.Dropout(0.5),
                      tf.keras.layers.Conv2D(filters=512, kernel_size=7, padding='same',activation='relu'),
                      tf.keras.layers.Dropout(0.5),
                      tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='relu')])


    # Using a transposed conv (w/ s=2) to upscale `f5` into a 14 x 14 map
    # so it can be merged with features from `f4_conv1` obtained from `f4`
    self.f5_conv3_upsample = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2,
                                use_bias=False, padding='same', activation='relu')
    
    # We repeat the operation to merge `merge1` and `f3` into a 28 x 28 map:
    self.f3_conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same',
                      activation="relu")
    
    self.merge_upsample = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2,
                                use_bias=False, padding='same', activation='relu')
    
    # Final layer
    self.final = tf.keras.layers.Conv2DTranspose(filters=31, kernel_size=16, strides=8,
                              padding='same', activation="softmax")
    
  def call(self, inputs):

    f3, f4, f5 = self.down_stack(inputs)

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


class DeepLabV3(tf.keras.Model):

    def __init__(self):

      super().__init__()

      # Feature extractor
      base_model = tf.keras.applications.resnet.ResNet101(include_top=False, input_shape=[128, 128, 3], weights='imagenet')

      self.down_stack = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer("conv3_block2_1_relu").output)

      self.cv1 = tf.keras.layers.Conv2D(256, 1, activation="relu")
      self.upsampling_v1 = tf.keras.layers.UpSampling2D(interpolation="bilinear")

      self.atrous_block_1 = tf.keras.layers.Conv2D(256, 1, padding='same', activation="relu")
      self.atrous_block_6 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=6, activation="relu")
      self.atrous_block_12 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=12, activation="relu")
      self.atrous_block_18 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=18, activation="relu")

      self.atrous_out = tf.keras.layers.Conv2D(256, 1, activation="relu")

      self.upsampling_v2 = tf.keras.layers.UpSampling2D(4, interpolation="bilinear")


      self.last_layer = tf.keras.layers.Conv2D(31, 1, activation="softmax")


    def call(self, inputs):

      image_features = self.down_stack(inputs)

      # AtrousSpatialPyramidPoolingModule starts

      image_features = self.cv1(image_features)
      image_features = self.upsampling_v1(image_features)

      atrous_block_1 = self.atrous_block_1(image_features)
      atrous_block_6 = self.atrous_block_6(image_features)
      atrous_block_12 = self.atrous_block_12(image_features)
      atrous_block_18 = self.atrous_block_18(image_features)

      net = tf.concat([image_features, atrous_block_1, atrous_block_6, atrous_block_12, atrous_block_18], axis=3)

      net = self.atrous_out(net)
      # AtrousSpatialPyramidModuleEnds

      x = self.upsampling_v2(net)

      outputs = self.last_layer(x)

      return outputs
    
class DeepLabV3_plus(tf.keras.Model):

    def __init__(self):

        super().__init__()

        base_model = tf.keras.applications.resnet.ResNet101(include_top=False, input_shape=[128, 128, 3], weights='imagenet')

        self.down_stack = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer("conv3_block2_1_relu").output)

        self.glob_avg = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

        self.conv1 = tf.keras.layers.Conv2D(256, 1, activation="relu")
        self.upsample = tf.keras.layers.UpSampling2D(interpolation="bilinear")

        self.atrous_block_1 = tf.keras.layers.Conv2D(256, 1, padding='same', activation="relu")
        self.atrous_block_6 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=6, activation="relu")
        self.atrous_block_12 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=12, activation="relu")
        self.atrous_block_18 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=18, activation="relu")

        self.atrous_out = tf.keras.layers.Conv2D(256, 1, activation="relu")

        self.conv2 = tf.keras.layers.Conv2D(256, 1, activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(48, 1, activation="relu")

        self.final_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, activation="relu"),
            tf.keras.layers.Conv2D(256, 3, activation="relu"),
            tf.keras.layers.UpSampling2D(interpolation="bilinear"),
            tf.keras.layers.Conv2D(31, 8, activation="softmax")
        ])


    def call(self, inputs):

        encoder_features = self.down_stack(inputs)

        image_features = self.glob_avg(encoder_features)

        image_features = self.conv1(image_features)
        image_features = self.upsample(image_features)

        atrous_block_1 = self.atrous_block_1(image_features)
        atrous_block_6 = self.atrous_block_6(image_features)
        atrous_block_12 = self.atrous_block_12(image_features)
        atrous_block_18 = self.atrous_block_18(image_features)

        net = tf.concat([image_features, atrous_block_1, atrous_block_6, atrous_block_12, atrous_block_18], axis=3)
        x = self.atrous_out(net)
        x = self.conv2(x)
        decoder_features = self.upsample(x)

        encoder_features = self.conv3(encoder_features)

        net = tf.concat([encoder_features, decoder_features], axis=3)


        outputs = self.final_block(net)

        return outputs



