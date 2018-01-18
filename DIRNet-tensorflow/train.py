import tensorflow as tf
from models import DIRNet
from config import get_config
from data import DIRNetDatahandler
from ops import mkdir


def main():
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    config = get_config(is_train=True)
    mkdir(config.tmp_dir)
    mkdir(config.ckpt_dir)

    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    # reg.restore(config.ckpt_dir)
    dh = DIRNetDatahandler( config=config)

    for i in range(config.iteration):
        # create new random batch
        batch_x, batch_y = dh.sample_pair(config.batch_size)

        # run sess => minimize loss
        loss = reg.fit(batch_x, batch_y)

        print("iter {:>6d} : {}".format(i + 1, loss))

        if (i + 1) % config.checkpoint_distance == 0:
            # reg.deploy(config.tmp_dir, batch_x, batch_y)
            reg.save(config.ckpt_dir)


if __name__ == "__main__":
    main()
