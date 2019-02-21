import os
import cv2


def load_image(name, conf):
    path = os.path.join(conf['train']['data-folder'], 'images', name)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    width = conf['models']['road_seg_module']['image-size']['width']
    height = conf['models']['road_seg_module']['image-size']['height']
    img = cv2.resize(img, (width, height))
    
    return img


def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
