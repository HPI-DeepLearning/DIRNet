import tensorflow as tf
from WarpST import WarpST
from AffineST import AffineST
from ops import *
import scipy.misc


class CNN(object):
    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x):
        # localisation network
        with tf.variable_scope(self.name, reuse=self.reuse):
            x = conv2d(x, "conv1", 64, 3, 1,
                       "SAME", True, tf.nn.elu, self.is_train)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

            x = conv2d(x, "conv2", 128, 3, 1,
                       "SAME", True, tf.nn.elu, self.is_train)
            x = conv2d(x, "out1", 128, 3, 1,
                       "SAME", True, tf.nn.elu, self.is_train)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            x = conv2d(x, "out2", 2, 3, 1,
                       "SAME", False, None, self.is_train)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        if self.reuse is None:
            self.var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.train.Saver(self.var_list)
            self.reuse = True
        return x

    def save(self, sess, ckpt_path):
        self.saver.save(sess, ckpt_path)

    def restore(self, sess, ckpt_path):
        self.saver.restore(sess, ckpt_path)


class DIRNet(object):
    def __init__(self, sess, config, name, is_train):
        self.sess = sess
        self.name = name
        self.is_train = is_train

        # image shape for grayscale images
        im_shape = [config.batch_size] + config.im_size + [1]
        # x => moving image
        self.x = tf.placeholder(tf.float32, im_shape)
        # y => fixed image
        self.y = tf.placeholder(tf.float32, im_shape)
        # x and y concatenated in color channel
        self.xy = tf.concat([self.x, self.y], 3)

        self.vCNN = CNN("vector_CNN", is_train=self.is_train)

        # calc v => featuremap from the localisation network
        self.v = self.vCNN(self.xy)
        self.z = None
        if config.use_AffineST:
            self.z = AffineST(self.x, self.v, config.im_size)
        else:
            # warp using deformable transformation
            # z contains the warped image
            self.z = WarpST(self.x, self.v, config.im_size)

        if self.is_train:
            # calc normalized cross correlation
            self.loss = ncc(self.y, self.z)
            # self.loss = mse(self.y, self.z)

            self.optim = tf.train.AdamOptimizer(config.lr)
            self.train = self.optim.minimize(
                - self.loss, var_list=self.vCNN.var_list)

        # self.sess.run(
        #  tf.variables_initializer(self.vCNN.var_list))
        self.sess.run(tf.global_variables_initializer())

    def fit(self, batch_x, batch_y):
        _, loss = \
            self.sess.run([self.train, self.loss],
                          {self.x: batch_x, self.y: batch_y})
        return loss

    def calc_rmse(self, x, y):
        '''
        calculates the root mean squared error of two arrays
        :param x:
        :param y:
        :return:
        '''
        error = np.subtract(x, y)
        squared = np.square(error)
        avg = np.average(squared)
        rooted = np.sqrt(avg)
        return rooted

    def calc_rmse_all(self, x, y, dir_path,save_images):
        '''
        calculates the rmse for all images in dir_path
        and prints the mean  rmse for all (x,z) and (x,y)
        :param x: numpy array of all moving images with dim-4
        :param y: numpy array of all fixed images with dim-4
        :param dir_path: directory from which to load the images
        :param save_images: saves all images if true
        :return: rmse for all (x,y) and (x,z) as tuple ((x,y),(x,z))
        '''
        rmse_original_res = 0
        rmse_registered_res = 0
        counter = 0
        for i in range(x.shape[0]):

            # calc transformed image
            z = self.sess.run(self.z,
                              {self.x: np.expand_dims(x[i, :, :, :], 0), self.y: np.expand_dims(y[i, :, :, :], 0)})

            # demean images
            z = z - np.mean(z[0, :, :, 0])
            x_new = x[i, :, :, 0] - np.mean(x[i, :, :, 0])
            y_new = y[i, :, :, 0] - np.mean(y[i, :, :, 0])

            # calc rmse for (x,y) and (y,z)
            rmse_original = self.calc_rmse(y_new, x_new)
            rmse_registered = self.calc_rmse(x_new, z[0, :, :, 0])

            # filter outliers ?
            # if abs(rmse_registered-rmse_original)<3:

            counter += 1
            rmse_original_res += rmse_original
            rmse_registered_res += rmse_registered

            # save x,y and registered image
            if save_images:
                scipy.misc.imsave(dir_path + "/{:02d}_x.tif".format(i + 1), x[i, :, :, 0])
                scipy.misc.imsave(dir_path + "/{:02d}_y.tif".format(i + 1), y[i, :, :, 0])
                scipy.misc.imsave(dir_path + "/{:02d}_z.tif".format(i + 1), z[0, :, :, 0])
        return (rmse_original_res/counter),(rmse_registered_res/counter)


    def deploy(self, dir_path, x, y):
        '''
        saves 5 images a couple of randomly chosen input images
        :param x: numpy array of all moving images with dim-4
        :param y: numpy array of all fixed images with dim-4
        :param dir_path: directory from which to load the images
        '''
        z = self.sess.run(self.z, {self.x: x, self.y: y})
        for i in range(z.shape[0]):
            # demean images
            z_new = z[i, :, :, 0] - np.mean(z[i, :, :, 0])
            x_new = x[i, :, :, 0] - np.mean(x[i, :, :, 0])
            y_new = y[i, :, :, 0] - np.mean(y[i, :, :, 0])

            # create difference images and ignore low values y-x
            array = np.subtract(x_new, y_new)
            low_values_flags = array < .8
            array[low_values_flags] = 0
            array = array[:, :]

            scipy.misc.imsave(dir_path + "/{:02d}_x-y.tif".format(i + 1), array)
            scipy.misc.imsave(dir_path + "/{:02d}_x.tif".format(i + 1), x_new)
            scipy.misc.imsave(dir_path + "/{:02d}_y.tif".format(i + 1), y_new)
            scipy.misc.imsave(dir_path + "/{:02d}_z.tif".format(i + 1), z_new)

            # create difference images and ignore low values for z-y
            array = np.subtract(z_new, y_new)
            low_values_flags = array < .8
            array[low_values_flags] = 0
            scipy.misc.imsave(dir_path + "/{:02d}_z-y.tif".format(i + 1), array[:, :])

    def save(self, dir_path):
        self.vCNN.save(self.sess, dir_path + "/model.ckpt")

    def restore(self, dir_path):
        self.vCNN.restore(self.sess, dir_path + "/model.ckpt")
