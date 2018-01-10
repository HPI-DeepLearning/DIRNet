
import logging

from requests.api import post

logging.getLogger().setLevel(logging.DEBUG)
import mxnet as mx
import numpy as np
from PIL import Image
import scipy.misc
from os import listdir
from os.path import isfile, join
import gzip
import struct
from mxnet import nd, autograd
from os import listdir
from os.path import isfile, join
import scipy.ndimage as ndimage
import scipy.misc as misc



def get_mnist(mnistdir='../data/'):
    def read_data(label_url, image_url):
        with gzip.open(label_url) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype=np.int8)
        with gzip.open(image_url, 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32) / 255
        return label, image

    (train_lbl, train_img) = read_data(
        mnistdir + 'train-labels-idx1-ubyte.gz', mnistdir + 'train-images-idx3-ubyte.gz')
    (test_lbl, test_img) = read_data(
        mnistdir + 't10k-labels-idx1-ubyte.gz', mnistdir + 't10k-images-idx3-ubyte.gz')
    return {'train_data': train_img, 'train_label': train_lbl,
            'test_data': test_img, 'test_label': test_lbl}


def get_mnist_data_iterator_w_labels(mnistdir='../data/', digit=1):
    def get_iterator_single_digit(data, label):
        one_digit_indices = []  # Contains all indices with images depicting the digit
        for index in range(len(label)):  # There might be a faster way to do this
            if label[index] == digit:
                one_digit_indices.append(index)
        one_digit_data = data[one_digit_indices]
        one_digit_label = label[one_digit_indices]
        fixed_image = one_digit_data[np.random.randint(0, len(one_digit_label))]
        one_digit_fixed_image = []  # array of same length as above data array, but its the same img multiple times
        for _ in one_digit_data:
            one_digit_fixed_image.append(fixed_image)

        iterator = mx.io.NDArrayIter([one_digit_fixed_image, one_digit_data],
                                     [one_digit_label, one_digit_label],
                                     batch_size=1, shuffle=True)
        return iterator

    mnist = get_mnist(mnistdir)
    train_iter = get_iterator_single_digit(mnist['train_data'], mnist['train_label'])
    val_iter = get_iterator_single_digit(mnist['test_data'], mnist['test_label'])
    return train_iter, val_iter


def get_mnist_data_iterator_two_data_sources(mnistdir='../data/', digit=1):
    def get_iterator_single_digit(data, label):
        one_digit_indices = []  # Contains all indices with images depicting the digit
        for index in range(len(label)):  # There might be a faster way to do this
            if label[index] == digit:
                one_digit_indices.append(index)
        one_digit_data = data[one_digit_indices]
        one_digit_label = label[one_digit_indices]
        fixed_image = one_digit_data[np.random.randint(0, len(one_digit_label))]
        one_digit_fixed_image = []  # array of same length as above data array, but its the same img multiple times
        for _ in one_digit_data:
            one_digit_fixed_image.append(fixed_image)
        data = {'data_fixed': one_digit_fixed_image, 'data_moving': one_digit_data}
        iterator = mx.io.NDArrayIter(data, batch_size=1, shuffle=True)
        return iterator

    mnist = get_mnist(mnistdir)
    train_iter = get_iterator_single_digit(mnist['train_data'], mnist['train_label'])
    val_iter = get_iterator_single_digit(mnist['test_data'], mnist['test_label'])
    return train_iter, val_iter


def get_mnist_data_iterator(mnistdir='./data/', digit=1):
    def get_iterator_single_digit(data, label):
        one_digit_indices = []  # Contains all indices with images depicting the digit
        for index in range(90):  # There might be a faster way to do this
        #for index in range(len(label)):  # There might be a faster way to do this
            if label[index] == digit:
                one_digit_indices.append(index)
        one_digit_data = data[one_digit_indices]
        #one_digit_label = label[one_digit_indices]
        # fixed_image = one_digit_data[np.random.randint(0, len(one_digit_label))]
        #data = {'data_fixed': one_digit_fixed_image, 'data_moving': one_digit_data}
        iterator = mx.io.NDArrayIter(one_digit_data, batch_size=1, shuffle=True)
        return iterator

    mnist = get_mnist(mnistdir)
    train_iter = get_iterator_single_digit(mnist['train_data'], mnist['train_label'])
    val_iter = get_iterator_single_digit(mnist['test_data'], mnist['test_label'])
    return train_iter, val_iter


def find_moving_img(arr, start_idx, fixed_name):
    # patient035_frame01.nz.10.png
    patient_id = fixed_name[7:10]
    slice_id = fixed_name[22:24]
    for i in range(len(arr)):
        idx = (i + start_idx) % len(arr)  # iterate through the whole array but dont start at 0
        moving_name = arr[idx]
        if patient_id == moving_name[7:10] and slice_id == moving_name[22:24]:
            return moving_name
    return None


def read_cardio_dirs_to_ndarray(path_fixed, path_moving, shape):
    '''
    Reads the fixed (ED) cardio images, looks for a corresponding moving image (ES) by name (same patient, same slice)
    and returns an array containing them
    :param path_fixed: path to ED
    :param path_moving: path to ES
    :param shape: target shape, will rescale all images to the same size
    :return: an array of shape (amount_pairs, 2, shape[0], shape[1]) fixed is in [idx][0] and mov in [idx][1]
    '''


    onlyfiles_fixed = [f for f in listdir(path_fixed) if isfile(join(path_fixed, f))]
    onlyfiles_moving = [f for f in listdir(path_moving) if isfile(join(path_moving, f))]
    # out_fix = np.empty(shape=(shape[1], shape[2]))
    # out_mov = np.empty(shape=(shape[1], shape[2]))
    arrays_fix = []
    arrays_mov = []
    for i, fixed in enumerate(onlyfiles_fixed):
        if fixed.endswith('.png'):
            moving = find_moving_img(onlyfiles_moving, i, fixed)
            assert moving is not None

            abspath = join(path_fixed, fixed)
            pic_fix = ndimage.imread(abspath, flatten=True)
            pic_fix = misc.imresize(pic_fix, (shape[0], shape[1]))

            abspath = join(path_moving, moving)
            pic_mov = ndimage.imread(abspath, flatten=True)
            pic_mov = misc.imresize(pic_mov, (shape[0], shape[1]))
            arrays_fix.append(np.stack([pic_fix, pic_mov]))
            #arrays_mov.append(pic_mov)
    return arrays_fix


def ncc(x, y):
    # mean_x = tf.reduce_mean(x, [1, 2, 3], keep_dims=True)
    # mean_y = tf.reduce_mean(y, [1, 2, 3], keep_dims=True)
    # mean_x2 = tf.reduce_mean(tf.square(x), [1, 2, 3], keep_dims=True)
    # mean_y2 = tf.reduce_mean(tf.square(y), [1, 2, 3], keep_dims=True)
    # stddev_x = tf.reduce_sum(tf.sqrt(
    #     mean_x2 - tf.square(mean_x)), [1, 2, 3], keep_dims=True)
    # stddev_y = tf.reduce_sum(tf.sqrt(
    #     mean_y2 - tf.square(mean_y)), [1, 2, 3], keep_dims=True)
    # return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))
    mean_x = mx.symbol.mean(data=x, axis=(1, 2, 3), keepdims=True)
    mean_y = mx.symbol.mean(data=y, axis=(1, 2, 3), keepdims=True)
    mean_x2 = mx.symbol.mean(mx.symbol.square(x), (1, 2, 3), keepdims=True)
    mean_y2 = mx.symbol.mean(mx.symbol.square(y), (1, 2, 3), keepdims=True)
    stddev_x = mx.symbol.sum(mx.symbol.sqrt(
        mean_x2 - mx.symbol.square(mean_x)), (1, 2, 3), keepdims=True)
    stddev_y = mx.symbol.sum(mx.symbol.sqrt(
        mean_y2 - mx.symbol.square(mean_y)), (1, 2, 3), keepdims=True)
    top = mx.symbol.broadcast_sub(x, mean_x) * (mx.symbol.broadcast_sub(y, mean_y))
    # return -mx.symbol.mean(mx.symbol.broadcast_div(top, mx.symbol.broadcast_sub((stddev_x * stddev_y), 0.1)))
    return -mx.symbol.mean(mx.symbol.broadcast_div((top + 0.1), ((stddev_x * stddev_y) + 0.1)))


def rmse(x, y):
    error = mx.symbol.broadcast_sub(x, y)
    squared = mx.symbol.square(error)
    avg = mx.symbol.mean(squared)
    rooted = mx.symbol.sqrt(avg)
    return rooted


def printNumpyArray(a, thresh=0.5):
    for i in range(len(a)):
        linestr = ''
        for k in range(len(a[0])):
            if a[i][k] > thresh:
                linestr += 'X'
            else:
                linestr += '_'
        print(linestr)


def saveArrayAsImg(array, filepath='./imgfromarray.png'):
    #print(type(array))
    #print(np.shape(array))
    #im = Image.fromarray(array)
    #im.save(filepath)
    scipy.misc.imsave(arr=array, name=filepath)

def printNontZeroGradients(grads, thresh=0):
    print("Gradient arrays that contain non-zero values:")
    for key in grads.keys():
        allZero = True
        for v in np.nditer(grads[key].asnumpy()):
            if v > thresh:
                allZero = False
                break
        if not allZero:
            print('\t ' + key)

def printNaNGradients(grads, thresh=0):
    print("Gradient arrays that contain non-zero values:")
    for key in grads.keys():
        hasNan = False
        for v in np.nditer(grads[key].asnumpy()):
            if np.isnan(v):
                hasNan = True
                break
        if hasNan:
            print('\t ' + key + ' has NaN values!')


def create_imglist(root_path, pathout=''):
    onlyfiles = [f for f in listdir(root_path) if isfile(join(root_path, f))]
    str_out = ''
    i = 0
    for filename in onlyfiles:
        # Format: Tab separated record of index, one or more labels and relative_path_from_root.
        i += 1
        if filename.endswith('.png'):
            str_out += str(i) + '\t1\t/\n'
    if pathout == '':
        pathout = root_path+'/imglist.txt'
    with open(pathout, 'w') as f:
        f.write(str_out)


def pure_batch_norm(X, gamma, beta, eps = 2e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        # normalize
        X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0, 2, 3))
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    return out