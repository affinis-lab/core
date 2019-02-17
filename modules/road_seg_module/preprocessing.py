import cv2
import numpy as np
import matplotlib.pyplot as plt


def fun(img, image_size):
    x = cv2.resize(
        src=img[300:, :, :],
        dsize=image_size,
        interpolation=cv2.INTER_NEAREST
    )
    return np.expand_dims(x, 0)
