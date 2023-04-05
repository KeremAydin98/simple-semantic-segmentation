from preprocessing import *
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import Augment, U_net
import config
import os

def display(display_list):

    plt.figure(figsize=(15,15))

    title = ['Input image', 'True mask', 'Predicted Mask']

    for i in range(len(display_list)):

        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    
    plt.show()

def show_predictions(dataset, model):

    for img, mask in dataset.take(2):

        sample_img, sample_mask = img[0], mask[0]

        predictions = model.predict(sample_img)

        predictions = tf.argmax(predictions, -1)
        predictions = tf.expand_dims(predictions)[0]

        display([sample_img, sample_mask, predictions])


# Extract the dataset from zipfile
zip_ref = zipfile.ZipFile("archive.zip")
zip_ref.extractall()
zip_ref.close()

# Load images and masks
image_paths = os.listdir(config.image_path)
mask_paths = os.listdir(config.mask_path)
image_paths = [config.image_path + image_path for image_path in image_paths]
mask_paths = [config.mask_path + mask_path for mask_path in mask_paths]
images, masks = load_images(image_paths, mask_paths)

# Parameters
batch_size = 8
buffer_size = 1000

# Split train and test data
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

# arrays to tf.data.dataset format
train_dataset = train_dataset.cache().shuffle(buffer_size).batch(batch_size).repeat().map(Augment()).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)

# Creating U-net semantic segmentation model
u_net = U_net()
u_net_model = u_net.create_model(3)
u_net_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

# Training the model
steps_per_epoch = len(train_images) // batch_size
validation_steps = len(test_images) // batch_size

history = u_net_model.fit(train_dataset, epochs=10, steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps, validation_data=test_dataset, verbose=2)

# Show predictions 
show_predictions(test_dataset, u_net)







