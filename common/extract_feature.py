import cv2
import h5py as hp
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from keras.applications.resnet50 import preprocess_input

# save
from configure import data_save_dir

casia_train_total = (25 * 2048 * 4) * (20 * 12 * 5 * 10) / (1024 ** 3)
print('Casia-Fasd: {:.02f}G'.format(casia_train_total))

replay_train_total = (25 * 2048 * 4) * (480 * 5 * 10) / (1024 ** 3)
print('Replay-Attack: {:.02f}G'.format(replay_train_total))


def Extractor():
    from keras.applications import ResNet50
    cnn = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    cnn.trainable = False
    return cnn


def read_video(file):
    cap = cv2.VideoCapture(file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    return np.asarray(frames).astype('float32')


def extract(nb_feature=2048):
    dataset = ['casia', 'replayattack']
    datatype = ['train', 'devel', 'test']
    factors = (1.8, 2.0, 2.2)

    model = Extractor()

    for ds, dt in product(dataset, datatype):

        save_path = '{}/{}-{}-lstm.h5'.format(data_save_dir, ds, dt)

        with hp.File(save_path, 'w') as h:
            for fct in factors:
                x = h.create_dataset(
                    '{}/data'.format(fct), (1, 25, nb_feature), maxshape=(None, 25, cfg.nb_feature),
                    chunks=(64, 25, nb_feature), dtype='float32'
                )
                y = h.create_dataset(
                    '{}/label'.format(fct), (1, 1), chunks=(64, 1), maxshape=(None, 1),
                    dtype='float32'
                )

                csv_path = 'csv/{}-{}-{}-lstm.csv'.format(ds, dt, fct)
                csv_data = pd.read_csv(csv_path, sep=',')

                length = len(csv_data)
                x.resize([length, 25, nb_feature])
                y.resize([length, 1])

                for i in tqdm(range(length)):
                    label = csv_data['label'][i]
                    video = csv_data['path'][i]
                    frames = read_video(video)
                    frames = frames[:, :, :, ::-1]
                    frames = preprocess_input(frames)
                    features = model.predict(frames)
                    x[i] = features.reshape((-1, 25, nb_feature))
                    y[i] = np.asarray([label]).reshape((-1, 1))
