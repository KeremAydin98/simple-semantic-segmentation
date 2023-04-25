import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import *
from models import *

# Load index labels
df = pd.read_csv("./CamVid/class_dict.csv")
df = df[["r", "g", "b"]]
label_dict = {str(np.array(label)):index for index, label in df.iterrows()}

images, masks = load_images("CamVid/train", label_dict)
val_images, val_masks = load_images("CamVid/val", label_dict)

train_dataset = tf.data.Dataset.from_tensor_slices((images, masks))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))

train_dataset = train_dataset.cache().batch(32).prefetch()
val_dataset = val_dataset.batch(32)

u_net_model = U_net()
model = u_net_model.create_model(1)

model.fit(train_dataset, epochs=10, validation_data=val_dataset)

