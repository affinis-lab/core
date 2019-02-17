import json
import os
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

from .model import build
from .generator import BatchGenerator
from .utils import r_square


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(dir, name):
    filename = os.path.join(dir, name)

    with open(filename, 'r') as f:
        data = json.load(f)

    return data


def load(name):
    path = os.path.join(BASE_DIR, 'pretrained', name)
    return load_model(path, compile=False)


def init(config):
    data = load_data(
        dir=config['train']['data-folder'],
        name=config['train']['data-file']
    )

    train_set, val_set = train_test_split(data, test_size=0.1, random_state=99)

    train_gen = BatchGenerator(config, train_set)
    val_gen = BatchGenerator(config, val_set)

    model = build(config, plot_core_model=config['plot_core_model'])
    model.load_weights()

    # optimizer = SGD(lr=1e-3, momentum=0.9, decay=0.0005)
    # optimizer = RMSprop(lr=1e-3,rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mae', r_square]
    )

    checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')

    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto',
        period = 1
    )

    batch_size = config['train']['batch_size']
    tensorboard = TensorBoard(
        batch_size=batch_size,
        update_freq=20 * batch_size
    )

    def train():
        history = model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            epochs=config['train']['nb_epochs'],
            workers=10,
            use_multiprocessing=True,
            shuffle=True,
            verbose=1,
            callbacks=[checkpoint, tensorboard],
        )

        return model, history

    def predict(x):
        return model.predict(x)[0]

    class Stub: pass
    res = Stub()
    res.train = train
    res.predict = predict

    return res

