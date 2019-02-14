import cv2
import matplotlib.pyplot as plt

from . import config


def fun(img):
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img)

    x = cv2.resize(
        src=img[300:, :, :],
        dsize=config.IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    plt.subplot(212)
    plt.imshow(x)
    plt.show()

    return x
