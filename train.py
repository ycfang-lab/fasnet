import os
import argparse
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint

from model import FasNet
from common.generator import train_gen, val_gen


def train(cfg):
    model = FasNet(cfg.input_num)
    prefix = os.path.join('weights', cfg.dataset)
    file_path = prefix + os.path.sep + 'weight.hdf5'
    if cfg.dataset == 'casia':
        steps_per_epoch = 459
        validation_steps = 114
    else:
        steps_per_epoch = 1518
        validation_steps = 151
    os.makedirs(prefix, exist_ok=True)
    es = EarlyStopping(monitor='val_acc', patience=10)
    cp = ModelCheckpoint(filepath=file_path, save_best_only=True, save_weights_only=True)

    model.fit_generator(
        generator=train_gen(cfg.dataset, cfg.input_num),
        validation_data=val_gen(cfg.dataset, cfg.input_num),
        steps_per_epoch=steps_per_epoch,
        callbacks=[es, cp],
        validation_steps=validation_steps,
        epochs=cfg.nb_epoch
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras: face-anti-spoofing')
    parser.add_argument('--nb_epoch', type=int, default=100)
    parser.add_argument('--input_num', type=int, default=1, choices=(1, 25))
    parser.add_argument('--dataset', type=str, default='casia', choices=('casia', 'replayattack'))
    cfg, _ = parser.parse_known_args()
    np.random.seed(9527)
    train(cfg)


