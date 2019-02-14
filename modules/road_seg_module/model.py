from keras.models import load_model
from segmentation_models.pspnet import PSPNet

from . import config


def build():
    model = PSPNet(
         backbone_name='vgg16',
         input_shape=config.INPUT_SHAPE,
         classes=config.NUM_CLASSES,
         activation='softmax',
         freeze_encoder=True,
    )
    model.load_weights(config.MODEL_PATH)
    return model
