from keras.models import load_model
import os
import numpy as np

from modules.traffic_light_module.yolo import YOLO, dummy_loss
from modules.traffic_light_module.preprocessing import load_image_predict
from modules.traffic_light_module.postprocessing import decode_netout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model(config):
    model = YOLO(
         config =config
    )
    model.load_weights(os.path.join(BASE_DIR, config['models']['traffic_light_module']['saved_model_name']))
    return model


def get_model_from_file(config):
    path = os.path.join(BASE_DIR, config['models']['traffic_light_module']['saved_model_name'])
    model = load_model(path, custom_objects={'custom_loss': dummy_loss})
    return model


def predict_with_model_from_file(config, model, image_path):
    image = load_image_predict(image_path, config['models']['traffic_light_module']['image_h'], config['models']['traffic_light_module']['image_w'])

    dummy_array = np.zeros((1, 1, 1, 1, config['models']['traffic_light_module']['max_obj']+4, 4))
    netout = model.predict([image, dummy_array])[0]

    boxes = decode_netout(netout=netout, anchors=config['models']['traffic_light_module']['anchors'],
                          nb_class=config['models']['traffic_light_module']['num_classes'],
                          obj_threshold=config['models']['traffic_light_module']['obj_thresh'],
                          nms_threshold=config['models']['traffic_light_module']['nms_thresh'])
    return boxes