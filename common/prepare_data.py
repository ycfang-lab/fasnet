import os
import cv2
import uuid
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product

from common.utils import get_file_list
from configure import clipped_videos_dir, clipped_videos_path_dir


def save_video(file_name, frames):
    fcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoWriter(file_name, fcc, 25.0, (224, 224))
    for i in range(len(frames)):
        x = cv2.resize(frames[i], dsize=(224, 224))
        cap.write(x)
    cap.release()


def label_video(attack_type):
    if attack_type == 'REAL_ACCESS':
        return 0
    if attack_type == 'PRINT_ATTACK':
        return 1
    if attack_type == 'PHOTO_ATTACK':
        return 2
    if attack_type == 'VIDEO_ATTACK':
        return 3


def calc_border(bbox, shape):
    l, t, r, b = [0] * 4
    x, y, w, h = bbox
    if x < 0:
        x, l = 0, -x
    if bbox[0] + w + l > shape[1]:
        r = bbox[0] + w + l - shape[1]
    if y < 0:
        y, t = 0, -y
    if bbox[1] + h + t > shape[0]:
        b = bbox[1] + h + t - shape[0]
    return (x, y, w, h), (l, t, r, b)


def calc_bbox(ffile, idxs, shape, datatype, factor=2.0):
    """ calculate bbox of each video clip"""
    bboxs = np.array([]).reshape(-1, 4)
    for i in idxs:
        bbox = ffile.get_face_roi(i)
        if bbox.any():
            bboxs = np.vstack((bboxs, bbox))
    if len(bboxs) == 0:
        return None, None
    x = bboxs[:, 0].min()
    y = bboxs[:, 1].min()
    w = (bboxs[:, 0] + bboxs[:, 2]).max() - x
    h = (bboxs[:, 1] + bboxs[:, 3]).max() - y
    x = int(x - w // factor)
    y = int(y - h // factor)
    w = int(w * factor)
    h = int(h * factor)
    bbox = [x, y, w, h]
    b, d = calc_border(bbox, shape)
    bboxs = np.array([b], dtype=np.int)
    dboxs = np.array([d], dtype=np.int)

    if datatype == 'train':
        for i, j in product([-1, 1], [-1, 1]):
            x = int(bbox[0] + i / 8 * bbox[2])
            y = int(bbox[1] + j / 8 * bbox[3])
            w = bbox[2]
            h = bbox[3]
            b, d = calc_border([x, y, w, h], shape)
            bboxs = np.vstack((bboxs, b))
            dboxs = np.vstack((dboxs, d))
    return bboxs, dboxs


def make_border(frames, border):
    """ make border of video clip """
    l, t, r, b = border
    color = [0, 0, 0]
    output = []
    for i in range(len(frames)):
        frame = cv2.copyMakeBorder(frames[i], t, b, l, r, cv2.BORDER_CONSTANT, value=color)
        output.append(frame)
    return np.array(output, dtype=np.uint8)


def video_resize(frames, size):
    output = np.zeros((25, 224, 224, 3), dtype=np.uint8)
    for i in range(len(frames)):
        frame = cv2.resize(frames[i], dsize=size)
        output[i:] = frame
    return output


def clip_video(vfile, ffile, datatype, factor):
    # (num, height, width, channels)
    frames = vfile.get_video_frames()
    shape = frames.shape[1:3]
    for i in range(1, len(frames) - 50, 50):
        idxs = np.linspace(i, i + 49, 25, dtype=np.int)
        # calculate boundingbox
        bboxs, dboxs = calc_bbox(ffile, idxs, shape, datatype, factor)
        if bboxs is None: continue
        for bbox, dbox in zip(bboxs, dboxs):
            x, y, w, h = bbox
            crop = make_border(frames[idxs], dbox)
            crop = crop[:, y:y + h, x:x + w]
            crop = video_resize(crop, (224, 224))
            yield crop
            # flip
            if datatype == 'train':
                yield crop[:, :, ::-1, :]


def prep_video(dataset, datatype, factor):
    vs, fs = get_file_list(dataset, datatype)
    prefix = os.path.join(clipped_videos_dir, dataset, datatype)

    tar_label = []
    tar_client = []
    tar_path = []

    for i in tqdm(range(len(vs))):

        vfile, ffile = vs[i], fs[i]
        attack = vfile.get_attack_type()
        client = vfile.get_client()
        label = label_video(attack)

        for v in clip_video(vfile, ffile, datatype, factor):
            save_dirs = os.path.join(prefix, str(factor), client)
            os.makedirs(save_dirs, exist_ok=True)
            if not os.path.exists(save_dirs):
                os.makedirs(save_dirs)
            save_path = save_dirs + '/' + str(uuid.uuid1()) + '.avi'
            save_video(save_path, v)

            tar_client.append(client)
            tar_label.append(label)
            tar_path.append(save_path)

    data_frame = pd.DataFrame({
        'client': tar_client, 'label': tar_label, 'path': tar_path
    })

    data_frame.to_csv('{}/{}-{}-{}.csv'.format(clipped_videos_path_dir, dataset, datatype, factor), index=False,
                      sep=',')


def prepare():
    dataset = ('casia', 'replayattack')
    datatype = ('train', 'devel', 'test')
    factors = (1.8, 2.0, 2.2)
    for i, j, k in product(dataset, datatype, factors):
        prep_video(i, j, k)
