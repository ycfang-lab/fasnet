import os
import cv2
import numpy as np

from configure import video_paths


class VideoFileName(object):
    def __init__(self, filename):
        self.file_name = filename

    def get_file_name(self):
        return self.file_name

    def get_video_frames(self):
        cap = cv2.VideoCapture(self.file_name)
        flag = cap.isOpened()
        frames = []
        while flag:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        # print(frames[0], frames[-1].shape)
        return np.asarray(frames)

    def get_attack_type(self):
        return self.attack_type

    def get_image_quality(self):
        return self.image_quality

    def get_client(self):
        return self.client


class CasiaVideoFileName(VideoFileName):
    def __init__(self, filename):
        super(CasiaVideoFileName, self).__init__(filename)
        self.client = filename.split('/')[-2]
        prefix, ext = filename.split('/')[-1].split('.')
        # parse image quality
        if prefix in ('1', '3', '5', '7'):
            self.image_quality = 'normal'
        elif prefix in ('2', '4', '6', '8'):
            self.image_quality = 'low'
        else:  # prefix in ('HR1', 'HR2', 'HR3', 'HR4')
            self.image_quality = 'high'
        # parse attack type
        if prefix in ('3', '4', 'HR_2'):
            self.attack_type = 'PRINT_ATTACK'
        elif prefix in ('5', '6', 'HR_3'):
            self.attack_type = 'PHOTO_ATTACK'
        elif prefix in ('7', '8', 'HR_4'):
            self.attack_type = 'VIDEO_ATTACK'
        else:  # prefix in ('1', '2', 'HR1')
            self.attack_type = 'REAL_ACCESS'

    @staticmethod
    def get_all_attack_types():
        return ['REAL_ACCESS', 'PRINT_ATTACK', 'PHOTO_ATTACK', 'VIDEO_ATTACK']

    @staticmethod
    def get_all_image_qualities():
        return ['normal', 'low', 'high']


class ReplayAttackVideoFileName(VideoFileName):
    def __init__(self, filename):
        super(ReplayAttackVideoFileName, self).__init__(filename)
        filename_split = filename.split('/')
        prefix, ext = filename_split[-1].split('.')
        if filename_split[-3] == 'attack':
            flag, media, client, session, image_quality, attack_type, scene = prefix.split('_')
            if media == 'print':
                self.attack_type = 'PRINT_ATTACK'
            elif attack_type == 'photo':
                self.attack_type = 'PHOTO_ATTACK'
            else:  # attack_type == 'video'
                self.attack_type = 'VIDEO_ATTACK'
        else:
            client, session, media, _, scene, _ = prefix.split('_')
            self.attack_type = 'REAL_ACCESS'
        self.image_quality = media
        self.scene = scene
        self.client = client
        self.media = media
        self.session = session

    def get_media(self):
        return self.media

    def get_scene(self):
        return self.scene

    def get_session(self):
        return self.session

    @staticmethod
    def get_all_attack_types():
        return ['REAL_ACCESS', 'PRINT_ATTACK', '', 'PHOTO_ATTACK', 'VIDEO_ATTACK']

    @staticmethod
    def get_all_image_qualities():
        return ['webcam', 'print', 'mobile', 'highdef']


class FaceFile(object):
    """Class for Processing face location file"""

    def __init__(self, filename):
        self.file_name = filename
        with open(filename, 'r') as f:
            lines = [k.strip() for k in f.readlines() if k.strip()]
            retval = np.zeros((len(lines), 5), dtype='int16')
            for i, line in enumerate(lines):
                s = line.split()
                for j in range(5):
                    retval[i, j] = int(s[j])
            self.retval = retval

    def get_file_name(self):
        """Get file name"""
        return self.file_name

    def get_mini_rect(self):
        """Get minimux scene"""
        temp = np.array([r for r in self.retval[:, 1:] if (r != 0).any()])
        x = temp[:, 0].min()
        y = temp[:, 1].min()
        w = (temp[:, 0] + temp[:, 2]).max() - x
        h = (temp[:, 1] + temp[:, 3]).max() - y
        return x, y, w, h

    def get_face_roi(self, index=0):
        """Get face position from """
        return self.retval[index, 1:]

    def get_frame_count(self):
        """Get frame count"""
        return len(self.retval)


def get_all_file(dirs):
    item_list = (os.path.join(dirs, item) for item in os.listdir(dirs))
    file_list = []
    for item in item_list:
        if os.path.isdir(item):
            file_list = file_list + get_all_file(item)
        else:
            file_list.append(item)
    return file_list


def get_file_list(dataset='casia', mode='train'):
    dirs = video_paths['{0}_{1}_{2}'.format(dataset, mode, 'face')]
    file_list = get_all_file(dirs)
    video_file_list = []
    face_file_list = []
    for file in file_list:
        video = file.replace('face-locations/', '').replace('.face', '.avi')
        if dataset == 'casia':
            video_file_list.append(CasiaVideoFileName(video))
        else:
            video = video.replace('.avi', '.mov')
            video_file_list.append(ReplayAttackVideoFileName(video))
        face_file_list.append(FaceFile(file))
    return video_file_list, face_file_list