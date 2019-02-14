from . import model
from . import preprocessing
from . import postprocessing


def get_output_type():
    return 'image'


def get_model():
    return model.build()


def get_preprocessing_fun():
    return preprocessing.fun


def get_postprocessing_fun():
    return postprocessing.fun
