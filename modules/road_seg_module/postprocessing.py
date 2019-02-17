import numpy as np


def fun(res):
    data = []
    converted = np.array([127, 127, 127])
    for row in res:
        r = [converted * np.argmax(pixel) for pixel in row]
        data.append(r)
    return np.array(data, dtype='uint8')
