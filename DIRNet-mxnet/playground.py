
import logging
import numpy as np
import gzip
import struct
import CustomNDArrayIter as customIter

logging.getLogger().setLevel(logging.DEBUG)
import mxnet as mx


def get_mnist(mnistdir='./data/'):
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


def get_mnist_data_iterator_w_labels(mnistdir='./data/', digit=1):
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


def get_mnist_data_iterator(mnistdir='./data/', digit=1):
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
        iterator = customIter.NDArrayIter(data, batch_size=1, shuffle=True)
        return iterator

    mnist = get_mnist(mnistdir)
    train_iter = get_iterator_single_digit(mnist['train_data'], mnist['train_label'])
    val_iter = get_iterator_single_digit(mnist['test_data'], mnist['test_label'])
    return train_iter, val_iter


def printNumpyArray(a, thresh=0.5):
    for i in range(len(a)):
        linestr = ''
        for k in range(len(a[0])):
            if a[i][k] > thresh:
                linestr += 'X'
            else:
                linestr += '_'
        print(linestr)

# helper function
def Init(key, arr):
    if "fc2_bias" in key:
        # initialize with identity transformation
        initial = np.array([[1., 1, 0], [1, 1., 0]])
        initial = np.array([[-0.00657603, -0.05496656,  0.14796074,  0.01942044,  0.04813603,  0.0093481 ]])
        initial = initial.astype('float32').flatten()
        arr[:] = initial
    elif "weight" in key:
        arr[:] = mx.random.uniform(-0.07, 0.07, arr.shape)
        # or
        # arr[:] = np.random.uniform(-0.07, 0.07, arr.shape)
    elif "gamma" in key:
        # for batch norm slope
        arr[:] = 1.0
    elif "bias" in key:
        arr[:] = 0
    elif "beta" in key:
        # for batch norm bias
        arr[:] = 0


data1 = mx.symbol.Variable('data1')
data2 = mx.symbol.Variable('data2')
net = mx.symbol.concat(data1, data2)
fc1 = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=6)
stnet = mx.sym.SpatialTransformer(data=data1, loc=net, target_shape=(28, 28), transform_type='affine',
                                  sampler_type="bilinear", name='SpatialTransformer')
cor = mx.sym.Correlation(data1=data1, data2=stnet, kernel_size=28, stride1=2, stride2=2, pad_size=0, max_displacement=0)
loss = mx.sym.MakeLoss(cor, normalization='batch')
# group fc1 and out together
group = mx.symbol.Group([mx.sym.BlockGrad(cor), mx.sym.BlockGrad(net), mx.sym.BlockGrad(stnet), loss])
#print group.list_outputs()




executor = group.simple_bind(ctx=mx.cpu(), data1=(1, 1, 28, 28), data2=(1, 1, 28, 28), label_shapes=None)
# get argument arrays
arg_arrays = executor.arg_arrays
# get grad arrays
grad_arrays = executor.grad_arrays
# get aux_states arrays. Note: currently only BatchNorm symbol has auxiliary states, which is moving_mean and moving_var
aux_arrays = executor.aux_arrays
# get outputs from executor
output_arrays = executor.outputs
# The sequence of arrays is in same sequence of symbol arguments
args = dict(zip(group.list_arguments(), arg_arrays))  # dict containing parameter names and values (i think)
grads = dict(zip(group.list_arguments(), grad_arrays))
outputs = dict(zip(group.list_outputs(), output_arrays))
aux_states = dict(zip(group.list_auxiliary_states(), aux_arrays))

# initialize parameters by uniform random numbers
for key, arr in args.items():
    Init(key, arr)
keys = group.list_arguments()
train_iter = get_mnist_data_iterator()[0]
for epoch in range(5):
    train_iter.reset()
    for batch in train_iter:
        outs = executor.forward(is_train=True, data1=batch.data[0], data2=batch.data[1])
        cor1 = executor.outputs[0]
        theta = executor.outputs[1]
        transformed = executor.outputs[2]
        transformed = transformed[0][0][:][:]
        printNumpyArray(batch.data[0][0][0])
        print('-------------------')
        printNumpyArray(transformed)
        loss = executor.outputs[3]
        executor.backward()  # compute gradients
        #for key in keys:  # update parameters
        #    customSGD(key, args[key], grads[key])
        #print('Epoch %d, Training cor %s grad %s' % (epoch, cor1, grad1))
