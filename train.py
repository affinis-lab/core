from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import json

from core_model import get_core_model
from preprocessing import load_data
from utils import BatchGenerator


def train(config):
    data = load_data(config['train']['train_annot_file'])

    train_generator = BatchGenerator(config, data)
    validation_generator = BatchGenerator(config, data)

    model = get_core_model(config, plot_core_model=False)

    # optimizer = SGD(lr=1e-3, momentum=0.9, decay=0.0005)
    # optimizer = RMSprop(lr=1e-3,rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
    )

    checkpoint = ModelCheckpoint(
        'checkpoints\\model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto',
        period = 1
    )

    model.summary()

    history = model.fit_generator(generator=train_generator,
                                       steps_per_epoch=len(train_generator),
                                       epochs=config['train']['nb_epochs'],
                                       verbose=1,
                                       validation_data=validation_generator,
                                       validation_steps=len(validation_generator),
                                       callbacks=[checkpoint],  #, tensorboard
                                       max_queue_size=8
                                       )

    with open('history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

