import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'road-seg-pretrained.h5'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

IMAGE_SIZE = (768, 192)
INPUT_SHAPE = (192, 768, 3)
NUM_CLASSES = 2