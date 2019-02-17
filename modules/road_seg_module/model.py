from keras.models import load_model
from segmentation_models.pspnet import PSPNet


def build(path):
    return load_model(path, compile=False)
