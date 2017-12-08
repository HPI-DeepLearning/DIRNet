
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



data1 = mx.symbol.Variable('data1')
data2 = mx.symbol.Variable('data2')
net = mx.symbol.concat(data1,data2)
fc1 = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=64)
out = mx.symbol.SoftmaxOutput(data=net, name='softmax')
cor = mx.sym.Correlation(data1=data1, data2=data2)
loss = mx.sym.MakeLoss(cor, normalization='batch')
# group fc1 and out together
group = mx.symbol.Group([mx.sym.BlockGrad(cor), loss])
#print group.list_outputs()

# You can go ahead and bind on the group
executor = group.simple_bind(ctx=mx.cpu(), data1=(1,1,28,28), data2=(1,1,28,28))
train_iter = get_mnist_data_iterator()[0]
for epoch in range(5):
    train_iter.reset()
    for batch in train_iter:
        outs = executor.forward(is_train=True, data1=batch.data[0], data2=batch.data[1])
        cor1 = executor.outputs[0]
        grad1 = executor.outputs[1]
        grad = executor.backward()  # compute gradients
        #for key in keys:  # update parameters
        #    customSGD(key, args[key], grads[key])
        print('Epoch %d, Training cor %s grad %s' % (epoch, cor1, grad1))
executor.forward()
executor.outputs[0] #will be value of fc1
executor.outputs[1] #will be value of softmax