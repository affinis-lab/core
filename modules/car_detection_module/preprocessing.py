import cv2
import numpy as np


def fun(img, image_size, max_objects):
    dummy_array = np.zeros((1, 1, 1, 1, max_objects, 4))
    x = cv2.resize(img, image_size) / 255
    x = np.expand_dims(x, 0)
    return [x, dummy_array]
