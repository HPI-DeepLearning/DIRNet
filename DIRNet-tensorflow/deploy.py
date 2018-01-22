import tensorflow as tf
from models import DIRNet
from config import get_config
from data import MNISTDataHandler
from ops import mkdir

def main():
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess = tf.Session(config=sess_config)
  config = get_config(is_train=True)
  # mkdir(config.result_dir)

  reg = DIRNet(sess, config, "DIRNet", is_train=False)
  # reg.restore(config.ckpt_dir)
  reg.restore('/home/adrian/PycharmProjects/DIRNet/DIRNet-tensorflow/ckpt')
  dh = MNISTDataHandler("MNIST_data", is_train=False,config=config)
  #
  # for i in range(10):
  #   result_i_dir = config.result_dir+"/{}".format(i)
  #   mkdir(result_i_dir)
  #
  #   batch_x, batch_y = dh.sample_pair(config.batch_size, i)
  #   reg.deploy(result_i_dir, batch_x, batch_y)
  reg.calc_rmse_all(dh.s_data, dh.d_data, '/home/adrian/PycharmProjects/DIRNet/DIRNet-tensorflow/tf_impl_rslt_all')
  # for i in range(833):
  #   result_i_dir = config.result_dir+"/{}".format(i)
  #   batch_x, batch_y = dh.get_pair_by_idx(i)
  #   reg.deploy(result_i_dir, batch_x, batch_y)

if __name__ == "__main__":
  main()
