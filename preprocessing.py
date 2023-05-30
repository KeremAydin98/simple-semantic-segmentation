import math, os
import numpy as np
import tensorflow as tf
from osgeo import gdal


def extract_data(root_paths, n_images=None):

  images = []
  masks = []

  iteration = 0

  for root_path in root_paths:

    for path in os.listdir(root_path):

      image_path = os.path.join(root_path, path)
      mask_path = os.path.join("L" + root_path[1:], path)

      smaller_image_path = image_path + "/img/"
      smaller_mask_path = mask_path + "/msk/"

      for file_name in os.listdir(smaller_image_path):

        msk_name = "MSK" + file_name[3:]

        sample_image = np.array(gdal.Open(smaller_image_path + file_name).ReadAsArray())
        sample_mask = np.array(gdal.Open(smaller_mask_path + msk_name).ReadAsArray())

        images.append(np.transpose(sample_image[:3, 224:448, 224:448], (1,2,0))[:, :, ::-1] / 255.0)
        masks.append(sample_mask[224:448, 224:448] - 1)

        iteration += 1

        if n_images:
          if iteration > n_images:
            images = np.array(images)
            masks = np.array(masks) 
            return images, masks

  images = np.array(images)
  masks = np.array(masks)

  return images, masks