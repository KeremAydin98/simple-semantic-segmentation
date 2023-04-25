import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import *
from models import *

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
df = pd.read_csv("./CamVid/class_dict.csv")
df = df[["r", "g", "b"]]
label_dict = {str(np.array(label)):index for index, label in df.iterrows()}

# Loading the dataset
train_images, train_masks = load_images("CamVid/train", label_dict)
val_images, val_masks = load_images("CamVid/val", label_dict)
test_images, test_masks = load_images("CamVid/test", label_dict)

# Parameters
batch_size = 32
buffer_size = 1000

# Converting numpy arrays to tf.data.Dataset 
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
train_dataset = train_dataset.cache().shuffle(buffer_size).batch(batch_size).repeat().map(Augment()).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size)

# Creating model
u_net_model = U_net()
model = u_net_model.create_model(1)
u_net_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

# Training the model
steps_per_epoch = len(train_images) // batch_size
validation_steps = len(val_images) // batch_size

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

# Show predictions 
show_predictions(test_dataset, model)






