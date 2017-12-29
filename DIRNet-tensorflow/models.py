import tensorflow as tf
from WarpST import WarpST
from ops import *
import scipy.misc

class CNN(object):
  def __init__(self, name, is_train,config):
    self.name = name
    self.is_train = is_train
    self.reuse = None

  def __call__(self, seqList):
    with tf.variable_scope(self.name, reuse=self.reuse):
      for i in range(0,len(seqList)):
        with tf.variable_scope(self.name+str(i), reuse=self.reuse):
          seqList[i] = conv2d(seqList[i], "conv1", 64, 3, 1,
            "SAME", True, tf.nn.elu, self.is_train)
          seqList[i] = tf.nn.avg_pool(seqList[i], [1,2,2,1], [1,2,2,1], "SAME")

          seqList[i] = conv2d(seqList[i], "conv2", 128, 3, 1,
            "SAME", True, tf.nn.elu, self.is_train)
          seqList[i] = conv2d(seqList[i], "out1", 128, 3, 1,
             "SAME", True, tf.nn.elu, self.is_train)
          seqList[i] = tf.nn.avg_pool(seqList[i], [1,2,2,1], [1,2,2,1], "SAME")
          seqList[i] = conv2d(seqList[i], "out2", 2, 3, 1,"SAME", False, None, self.is_train)

      multiply_weights = tf.get_variable("multiply_weights", [len(seqList)])
      # init output
      output=tf.multiply(seqList[i],multiply_weights[0])
      # for i in range(1,len(seqList)):
      #     # concat ouputs from different layers and multiplying them with their weights
      #     output=tf.add(output,tf.multiply(seqList[i],multiply_weights[i]))

    if self.reuse is None:
      self.var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(self.var_list)
      self.reuse = True
    print(str(output.get_shape())+"thats our shape")
    return seqList[0]

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)

class DIRNet(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train
    self.im_list=[]

    # moving / fixed images
    im_shape = [config.sequence_length] + config.im_size+[3]
    self.x = tf.placeholder(tf.float32, im_shape)
    self.y = tf.placeholder(tf.float32, im_shape)
    self.xy = tf.concat([self.x, self.y], 3)

    for i in range(0,self.xy.get_shape()[0]):
        self.im_list.append(tf.expand_dims(self.xy[i,:,:,:],0))
        print(self.im_list[i].get_shape())

    # what would be better ->
    # multiple images should be concatenated in first dimension -> instead of batches we get sequences
    # hard part is how to assign result of the conv nets ... maybe it has an easy solution, that actually uses the gpu


    self.vCNN = CNN("vector_CNN", is_train=self.is_train, config = config)
    print("x vlaue"+str(self.x.get_shape()))
    # vector map & moved image
    self.v = self.vCNN(self.im_list)

    param_buff=tf.cast(tf.identity(self.v),dtype=tf.float32)
    for i in range(1,config.sequence_length):
        param_buff=tf.concat([param_buff,tf.identity(self.v)],0)
    print(param_buff.get_shape())
    # print("concat dims "+str(param_buff.get_shape()))
    self.z = WarpST(self.x, param_buff, config.im_size)

    if self.is_train :
      self.loss = ncc(self.y, self.z)
      #self.loss = mse(self.y, self.z)

      self.optim = tf.train.AdamOptimizer(config.lr)
      self.train = self.optim.minimize(
        - self.loss, var_list=self.vCNN.var_list)

    #self.sess.run(
    #  tf.variables_initializer(self.vCNN.var_list))
    self.sess.run(tf.global_variables_initializer())

  def fit(self, batch_x, batch_y):
    _, loss = \
      self.sess.run([self.train, self.loss],
      {self.x:batch_x, self.y:batch_y})
    return loss

  def deploy(self, dir_path, x, y):
    z = self.sess.run(self.z, {self.x:x, self.y:y})
    for i in range(z.shape[0]):
      scipy.misc.imsave(dir_path+"/{:02d}_x.tif".format(i+1), x[i,:,:,:])
      scipy.misc.imsave(dir_path+"/{:02d}_y.tif".format(i+1), y[i,:,:,:])
      scipy.misc.imsave(dir_path+"/{:02d}_z.tif".format(i+1), z[i,:,:,:])

  def save(self, dir_path):
    self.vCNN.save(self.sess, dir_path+"/model.ckpt")

  def restore(self, dir_path):
    self.vCNN.restore(self.sess, dir_path+"/model.ckpt")
