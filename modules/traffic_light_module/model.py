from keras.models import load_model


def build(path):
    return load_model(path, compile=False)
