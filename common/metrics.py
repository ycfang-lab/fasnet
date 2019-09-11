"""
"""

import numpy as np
import keras.backend as K


def calc_err_rate(y_true, y_pred):
    """ calculate error rate
    """
    y_true_pos = np.asarray(y_true).reshape((-1, 1))
    y_pred_pos = np.asarray(y_pred).reshape((-1, 1))

    y_true_neg = 1 - y_true_pos
    y_pred_neg = 1 - y_pred_pos

    tp = np.sum(y_true_pos * y_pred_pos)
    tn = np.sum(y_true_neg * y_pred_neg)
    fp = np.sum(y_true_neg * y_pred_pos)
    fn = np.sum(y_true_pos * y_pred_neg)

    frr = fn / max((fn + tp), K.epsilon())
    far = fp / max((fp + tn), K.epsilon())

    acc = (tp + tn) / max((tp + tn + fp + fn), K.epsilon())
    return far, frr, acc


def calc_hter_by_class(y_true, y_pred):
    """ calculate detection error tradeoff with class
    """
    far, frr, acc = calc_err_rate(y_true, y_pred)
    hter = (frr + far) / 2
    return hter, acc


def calc_hter_by_proba(y_true, y_pred, thd=0.5):
    """ calculate detection error with classification probablity
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.clip(y_true, 0, 1) >= thd
    y_pred = np.clip(y_pred, 0, 1) >= thd
    far, frr, acc = calc_err_rate(y_true, y_pred)
    hter = (far + frr) / 2
    return hter, acc


def calc_det_curve(y_true, y_pred):
    """ calc det threshold
    """
    y_true = np.asarray(y_true).reshape((-1, 1))
    y_pred = np.asarray(y_pred).reshape((-1, 1))
    thd_idx = np.argsort(y_pred, axis=0)
    far_list, frr_list = [], []
    for idx in thd_idx:
        _y_true = y_true >= y_pred[idx]
        _y_pred = y_pred >= y_pred[idx]
        far, frr, _ = calc_err_rate(_y_true, _y_pred)
        far_list.append(far)
        frr_list.append(frr)
    return np.asarray(far_list), np.asarray(frr_list), y_pred[thd_idx].reshape(-1)


def calc_eer_thd(far, frr, thd):
    """ calculate equal error rate
    """
    far = np.asarray(far).reshape(-1)
    frr = np.asarray(frr).reshape(-1)
    idx = np.argmin(np.abs(far - frr))
    return frr[idx], thd[idx]
