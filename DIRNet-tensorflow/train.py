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
  mkdir(config.tmp_dir)
  mkdir(config.ckpt_dir)

  reg = DIRNet(sess, config, "DIRNet", is_train=True)
  # reg.restore(config.ckpt_dir)
  print('start reading data')
  dh = MNISTDataHandler("MNIST_data", is_train=True, config=config)


  for epoch in range(17):
    loss_sum = 0
    for i in range(833):
      batch_x, batch_y = dh.get_pair_by_idx(i)
      # loss = reg.fit((1, batch_x[0], batch_x[1], batch_x[2]),
      #                (1, batch_y[0], batch_y[1], batch_y[2]))
      loss = reg.fit(batch_x, batch_y)
      loss_sum += loss
    print("iter {:>6d} : {}".format(epoch, loss_sum/833))

    if (epoch+1) % 5 == 0:
    # if (epoch+1) % config.checkpoint_distance == 0:
      # reg.deploy(config.tmp_dir, batch_x, batch_y)
      reg.save(config.ckpt_dir)

if __name__ == "__main__":
  main()
