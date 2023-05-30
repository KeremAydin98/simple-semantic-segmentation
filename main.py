import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import *
from models import *
import glob
import zipfile


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

# Load index labels
zip_ref = zipfile.ZipFile("flair.zip")
zip_ref.extractall()
zip_ref.close()

zip_ref = zipfile.ZipFile("labels.zip")
zip_ref.extractall()
zip_ref.close()

# Parameters
batch_size = 32
buffer_size = 1000

# Loading the datasets
train_images, train_masks = extract_data(['D075_2021','D076_2019', 'D083_2020'])
test_images, test_masks = extract_data(['D085_2019'])

# arrays to tf.data.dataset format
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

train_dataset = train_dataset.cache().shuffle(buffer_size).batch(batch_size).repeat().map(Augment()).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Creating model
model = U_net()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

# Training the model
steps_per_epoch = len(train_images) // batch_size
validation_steps = len(test_images) // batch_size


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_dataset, epochs=100, steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps, validation_data=test_dataset, callbacks=[early_stopping])


# The '.h5' extension indicates that the model should be saved to HDF5.
model.save_weights('u_net.h5')

# Show predictions 
show_predictions(test_dataset, model)






