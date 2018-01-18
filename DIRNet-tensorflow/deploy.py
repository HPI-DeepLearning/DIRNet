import tensorflow as tf
from models import DIRNet
from config import get_config
from data import DIRNetDatahandler
from ops import mkdir


def main():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    config = get_config(is_train=False)
    mkdir(config.result_dir)

    reg = DIRNet(sess, config, "DIRNet", is_train=False)
    reg.restore(config.ckpt_dir)
    dh = DIRNetDatahandler( config=config)

    print(reg.calc_rmse_all(dh.s_data, dh.d_data, config.result_dir + "/",save_images=False))

    # to use the deploy func from models

    # for i in range(10):
    #   result_i_dir = config.result_dir+"/{}".format(i)
    #   mkdir(result_i_dir)
    #
    #   batch_x, batch_y = dh.sample_pair(config.batch_size, i)
    #   reg.deploy(result_i_dir, batch_x, batch_y)


if __name__ == "__main__":
    main()
