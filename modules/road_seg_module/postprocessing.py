import numpy as np


KITTI_ROAD_ENCODING_INVERTED = {
    0: [  0,   0,   0], # void
    1: [255, 255, 255], # drivable
}


def fun(res):
    data = []
    for row in res:
        r = []
        for pixel in row:
            index = np.argmax(pixel)
            r.append(KITTI_ROAD_ENCODING_INVERTED[index])
        data.append(r)
    return np.array(data, dtype='uint8')
