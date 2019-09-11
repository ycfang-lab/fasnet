import os
import argparse
import numpy as np

from model import FasNet
from common.visualize import plot_det_curve
from common.generator import devel_gen, test_gen
from common.metrics import calc_hter_by_proba, calc_det_curve, calc_eer_thd


def evaluate(cfg):
    os.makedirs('./image', exist_ok=True)
    model = FasNet(cfg.input_num, cfg.nb_feature)
    model.load_weights(cfg.checkpoint_path)
    # devel
    y_true_devel, y_pred_devel = [], []
    for x_batch, y_batch in devel_gen(cfg.dataset, cfg.input_num):
        pred = model.predict(x_batch)
        y_true_devel.append(y_batch)
        y_pred_devel.append(pred)
    y_true_devel = np.asarray(y_true_devel).reshape((-1, 1))
    y_pred_devel = np.asarray(y_pred_devel).reshape((-1, 1))
    print('{} devel samples'.format(y_true_devel.shape[0]))
    dev_far, dev_frr, dev_thd = calc_det_curve(y_true_devel, y_pred_devel)
    dev_eer, dev_thd = calc_eer_thd(dev_far, dev_frr, dev_thd)
    print('devel: EER: {:.04f}, THD: {:.04f}'.format(dev_eer, dev_thd))
    # plot det curve
    print('plot det on replay devel')
    plot_det_curve(dev_far, dev_frr, info='image/{}-devel'.format(cfg.dataset))

    y_true_test, y_pred_test = [], []
    for x_batch, y_batch in test_gen(cfg.dataset, cfg.input_num):
        pred = model.predict(x_batch)
        y_true_test.append(y_batch)
        y_pred_test.append(pred)
    y_true_test = np.asarray(y_true_test).reshape((-1, 1))
    y_pred_test = np.asarray(y_pred_test).reshape((-1, 1))
    print('{} test samples'.format(y_true_test.shape[0]))
    test_far, test_frr, test_thd = calc_det_curve(y_true_test, y_pred_test)
    # plot det curve
    print('plot det on replay test')
    plot_det_curve(test_far, test_frr, info='image/{}-test'.format(cfg.dataset))
    test_eer, test_thd = calc_eer_thd(test_far, test_frr, test_thd)
    print('test: EER: {:.04f}, THD: {:.04f}'.format(test_eer, test_thd))
    hter, acc = calc_hter_by_proba(y_true_test, y_pred_test, 0.5)
    print('test: O-Hter: {:.04f}, ACC: {:.04f}'.format(hter, acc))
    hter, acc = calc_hter_by_proba(y_true_test, y_pred_test, dev_thd)
    print('test: F-Hter: {:.04f}, ACC: {:.04f}'.format(hter, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras: face-anti-spoofing')
    parser.add_argument('--input_num', type=int, default=1, choices=(1, 25))
    parser.add_argument('--dataset', type=str, default='casia', choices=('casia', 'replayattack'))
    parser.add_argument('--checkpoint_path', type=str, required=True)
    cfg, _ = parser.parse_known_args()
    evaluate(cfg)
