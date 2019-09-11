import h5py as hp
import numpy as np


def train_gen(dataset, input_num=25):
    assert input_num in (1, 25)
    assert dataset in ('casia', 'replayattack')
    data_file = '-'.join(['h5/' + dataset, 'train', 'lstm.h5'])

    scale = (1.8, 2.0, 2.2)
    batch_size = 32
    while True:
        with hp.File(data_file, 'r') as h:
            for s in scale:
                x = h[str(s)]['data']
                y = h[str(s)]['label']
                if dataset == 'casia':
                    split = len(h[str(s)]['label']) // 5
                    if input_num == 25:
                        x = x[split:]
                        y = y[split:]
                    else:
                        x = x[split:, 0, :]
                        y = y[split:]
                        x = x.reshape(-1, 1, 2048)
                else:
                    if input_num == 25:
                        x = x[:]
                        y = y[:]
                    else:
                        x = x[:, 0, :]
                        y = y[:]
                        x = x.reshape(-1, 1, 2048)
                length = len(y)
                order = np.random.permutation(length)
                for i in range(length // batch_size):
                    idx = order[i * batch_size: (i + 1) * batch_size]
                    yield np.asarray(x[idx]), (np.asarray(y[idx]) == 0)
        if len(scale) == 1:
            break


def val_gen(dataset, input_num):
    assert input_num in (1, 25)
    assert dataset in ('casia', 'replayattack')
    if dataset == 'casia':
        data_file = '-'.join(['h5/' + dataset, 'train', 'lstm.h5'])
    else:
        data_file = '-'.join(['h5/' + dataset, 'devel', 'lstm.h5'])
    scale = (1.8, 2.0, 2.2)
    batch_size = 32
    while True:
        with hp.File(data_file, 'r') as h:
            for s in scale:
                x = h[str(s)]['data']
                y = h[str(s)]['label']
                if dataset == 'casia':
                    split = len(h[str(s)]['label']) // 5
                    if input_num == 25:
                        x = x[:split]
                        y = y[:split]
                    else:
                        x = x[:split, 0, :]
                        y = y[:split]
                        x = x.reshape(-1, 1, 2048)
                else:
                    if input_num == 25:
                        x = x[:]
                        y = y[:]
                    else:
                        x = x[:, 0, :]
                        y = y[:]
                        x = x.reshape(-1, 1, 2048)
                length = len(y)
                order = np.random.permutation(length)
                for i in range(length // batch_size):
                    idx = order[i * batch_size: (i + 1) * batch_size]
                    yield np.asarray(x[idx]), (np.asarray(y[idx]) == 0)


def devel_gen(dataset, input_num):
    assert input_num in (1, 25)
    assert dataset in ('casia', 'replayattack')
    if dataset == 'casia':
        data_file = '-'.join(['h5/' + dataset, 'train', 'lstm.h5'])
    else:
        data_file = '-'.join(['h5/' + dataset, 'devel', 'lstm.h5'])
    batch_size = 32
    with hp.File(data_file, 'r') as h:
        x = h['2.0']['data']
        y = h['2.0']['label']
        if dataset == 'casia':
            split = len(h['2.0']['label']) // 5
            if input_num == 25:
                x = x[:split]
                y = y[:split]
            else:
                x = x[:split, 0, :]
                y = y[:split]
                x = x.reshape(-1, 1, 2048)
        else:
            if input_num == 25:
                x = x[:]
                y = y[:]
            else:
                x = x[:, 0, :]
                y = y[:]
                x = x.reshape(-1, 1, 2048)
        length = len(y)
        order = np.array(range(length))
        for i in range(length // batch_size):
            idx = order[i * batch_size: (i + 1) * batch_size]
            yield np.asarray(x[idx]), (np.asarray(y[idx]) == 0)


def test_gen(dataset, input_num):
    assert input_num in (1, 25)
    assert dataset in ('casia', 'replayattack')
    data_file = '-'.join(['h5/' + dataset, 'test', 'lstm.h5'])
    batch_size = 32
    with hp.File(data_file, 'r') as h:
        x = h['2.0']['data']
        y = h['2.0']['label']
        if input_num == 25:
            x = x[:]
            y = y[:]
        else:
            x = x[:, 0, :]
            y = y[:]
            x = x.reshape(-1, 1, 2048)
        length = len(y)
        order = np.array(range(length))
        for i in range(length // batch_size):
            idx = order[i * batch_size: (i + 1) * batch_size]
            yield np.asarray(x[idx]), (np.asarray(y[idx]) == 0)
