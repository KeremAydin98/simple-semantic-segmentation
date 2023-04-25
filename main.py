import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import *
from models import *
import glob

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
df = pd.read_csv("./CamVid/class_dict.csv")[["r", "g", "b"]]
label_dict = {str(row.values): i for i, row in enumerate(df.values)}

# Parameters
batch_size = 32
buffer_size = 1000

# Loading the datasets
image_paths = glob.glob("./CamVid/train/*")
mask_paths = [p.replace("train", "train_labels") for p in image_paths]
train_dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
train_dataset = train_dataset.map(lambda x, y: preprocess(x, y, label_dict))
train_dataset = train_dataset.batch(batch_size)

image_paths = glob.glob("./CamVid/val/*")
mask_paths = [p.replace("val", "val_labels") for p in image_paths]
val_dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
val_dataset = val_dataset.map(lambda x, y: preprocess(x, y, label_dict))
val_dataset = val_dataset.batch(batch_size)

image_paths = glob.glob("./CamVid/test/*")
mask_paths = [p.replace("test", "test_labels") for p in image_paths]
test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
test_dataset = test_dataset.map(lambda x, y: preprocess(x, y, label_dict))
test_dataset = test_dataset.batch(batch_size)

# Creating model
u_net_model = U_net()
model = u_net_model.create_model(1)
u_net_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

# Training the model
steps_per_epoch = len(os.listdir("./CamVid/train/")) // batch_size
validation_steps = len(os.listdir("./CamVid/val/")) // batch_size

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

# Show predictions 
show_predictions(test_dataset, model)






