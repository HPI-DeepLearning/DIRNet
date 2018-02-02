import tensorflow as tf
from models import DIRNet,ResNet
from config import get_config
from data import DIRNetDatahandler
import numpy as np
from ops import mkdir


def main():
    tf.reset_default_graph()
    eval_resnet()
    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True
    # sess = tf.Session(config=sess_config)
    # config = get_config(is_train=True)
    # mkdir(config.result_dir)
    #
    # reg = DIRNet(sess, config, "DIRNet", is_train=False)
    # reg.restore(config.ckpt_dir)
    # dh = DIRNetDatahandler(config=config)
    #
    # # print(reg.calc_rmse_all(dh.s_data, dh.d_data, config.result_dir + "/",save_images=False))
    # batch_x, batch_y, batch_labels = dh.sample_pair(config.batch_size)
    # # prediction = reg.deploy_with_labels(batch_x, batch_y, batch_labels)
    # # print(str(prediction))
    #
    # amnt_pics = np.shape(dh.d_data)[0]
    # acc = 0
    # prev_x = np.empty(shape=(1, 222, 247))
    # amnt_eva = np.shape(dh.d_data_eval)[0]
    # for i in range(amnt_eva):
    #     batch_x, batch_y, batch_labels = dh.get_eval_pair_by_idx(i)
    #     if np.array_equal(prev_x, batch_x):
    #         print('weird')
    #     prev_x = batch_x
    #     # loss = reg.fit((1, batch_x[0], batch_x[1], batch_x[2]),
    #     #                (1, batch_y[0], batch_y[1], batch_y[2]))
    #     prediction = reg.deploy_with_labels(batch_x, batch_y, batch_labels)
    #     truth = int(batch_labels[0])
    #     # print("pred {} truth {}".format(prediction, truth))
    #     if prediction == truth:
    #         acc += 1
    # print("Acc: {0:.4f}".format( acc / amnt_eva))
    # # to use the deploy func from models
    #
    # # for i in range(10):
    # #   result_i_dir = config.result_dir+"/{}".format(i)
    # #   mkdir(result_i_dir)
    # #
    # #   batch_x, batch_y = dh.sample_pair(config.batch_size, i)
    # #   reg.deploy(result_i_dir, batch_x, batch_y)

def eval_resnet():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    config = get_config(is_train=True)
    mkdir(config.result_dir)

    reg = ResNet(sess, config, "DIRNet", is_train=False)
    reg.restore(config.ckpt_dir)
    dh = DIRNetDatahandler(config=config)

    # print(reg.calc_rmse_all(dh.s_data, dh.d_data, config.result_dir + "/",save_images=False))
    batch_x, batch_y, batch_labels = dh.sample_pair(config.batch_size)
    # prediction = reg.deploy_with_labels(batch_x, batch_y, batch_labels)
    # print(str(prediction))

    amnt_pics = np.shape(dh.d_data)[0]
    acc = 0
    prev_x = np.empty(shape=(1, 222, 247))
    amnt_eva = np.shape(dh.d_data_eval)[0]
    for i in range(amnt_eva):
        batch_x, batch_y, batch_labels = dh.get_eval_pair_by_idx(i)
        if np.array_equal(prev_x, batch_x):
            print('weird')
        prev_x = batch_x
        # loss = reg.fit((1, batch_x[0], batch_x[1], batch_x[2]),
        #                (1, batch_y[0], batch_y[1], batch_y[2]))
        prediction = reg.deploy_with_labels(batch_x, batch_y, batch_labels)
        print(prediction,"::", batch_labels[0])
        truth = int(batch_labels[0])
        # print("pred {} truth {}".format(prediction, truth))
        if prediction == truth:
            acc += 1
    print("Acc: {0:.4f}".format(acc / amnt_eva))
    # to use the deploy func from models

    # for i in range(10):
    #   result_i_dir = config.result_dir+"/{}".format(i)
    #   mkdir(result_i_dir)
    #
    #   batch_x, batch_y = dh.sample_pair(config.batch_size, i)
    #   reg.deploy(result_i_dir, batch_x, batch_y)

if __name__ == "__main__":
    main()
