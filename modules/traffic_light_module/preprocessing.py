import cv2
import numpy as np

def load_image_predict(image_path, image_h, image_w):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_h, image_w))
    image = image/255
    image = np.expand_dims(image, 0)

    return image
