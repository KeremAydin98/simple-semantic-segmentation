import matplotlib.pyplot as plt
import tensorflow as tf

def show_predictions(dataset, model):

  shuffled_dataset = dataset.shuffle(buffer_size=10000)

  for image, mask in shuffled_dataset.take(1):

    pred = model.predict(tf.expand_dims(image[0], 0))

    fig, ax = plt.subplots(1,3, figsize=(12,12))

    print(image[0].shape)
    print(mask[0].shape)
    print(tf.argmax(pred[0], -1).shape)

    ax[0].imshow(image[0])
    ax[1].imshow(mask[0])
    ax[2].imshow(tf.argmax(pred[0], -1))

  plt.show()