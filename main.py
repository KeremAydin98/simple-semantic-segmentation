from preprocessing import *
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import config
import os

# Extract the dataset from zipfile
zip_ref = zipfile.ZipFile("archive.zip")
zip_ref.extractall()
zip_ref.close()

# Load images and masks
image_paths = os.listdir(config.image_path)
mask_paths = os.listdir(config.mask_path)
images, masks = load_images(image_paths, mask_paths)

# Parameters
train_length = len(image_paths)
batch_size = 64
buffer_size = 1000
steps_per_epoch = train_length // batch_size

# Split train and test data
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2)




