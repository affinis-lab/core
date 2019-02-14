from . import model
from . import preprocessing


def get_output_type():
    return 'vector'


def get_model():
    return model.build()


def get_preprocessing_fun():
    return preprocessing.fun
