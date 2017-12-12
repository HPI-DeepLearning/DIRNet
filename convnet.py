'''
Implementation of the following paper:
https://arxiv.org/abs/1704.06065
'''
import logging

logging.getLogger().setLevel(logging.DEBUG)
import mxnet as mx
import numpy as np
import gzip
import struct
import CustomNDArrayIter as customIter
from collections import OrderedDict


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


def conv_net_regressor(image_shape, bn_mom=0.9):
    (nchannel, height, width) = image_shape
    # We have 2 data sources and concatenate them
    data_fixed = mx.sym.Variable(name='data_fixed')
    data_moving = mx.sym.Variable(name='data_moving')
    concat_data = mx.sym.concat(*[data_fixed, data_moving])
    batched = mx.sym.BatchNorm(data=concat_data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    # The number of kernels per layer can be of arbitrary size, but the number of kernels of the output layer is
    # determined by the dimensionality of the input images
    filter_list = [16, 32, 64, 128]
    # four alternating layers of 3 × 3 convolutions with 0-padding and 2 × 2 downsampling layers
    for i in range(4):
        if i == 0:
            body = mx.sym.Convolution(data=batched, num_filter=filter_list[i], kernel=(3, 3), stride=(1, 1), pad=(0, 0),
                                      no_bias=True, name="conv" + str(i))
        else:
            body = mx.sym.Convolution(data=body, num_filter=filter_list[i], kernel=(3, 3), stride=(1, 1), pad=(0, 0),
                                      no_bias=True, name="conv" + str(i))
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn' + str(i))
        # TO DO: the original authors use exponential linear units as activation
        body = mx.sym.Activation(data=body, act_type='relu', name='relu' + str(i))
        body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='avg')
    # Subsequently, three 1 × 1 convolutional layers are applied to make the ConvNet regressor fully convolutional
    for k in range(3):
        i = k + 4
        body = mx.sym.Convolution(data=body, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                  no_bias=True, name="conv" + str(i))
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn' + str(i))
        # TO DO: the original authors use exponential linear units as activation
        body = mx.sym.Activation(data=body, act_type='relu', name='relu' + str(i))
        body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='avg')
        #
        flatten = mx.sym.flatten(data=body)
        fc3 = mx.sym.FullyConnected(data=flatten, num_hidden=6)  # Todo: Initialize as identity
    # The Spatial Transformer performs a affine transformation to the moving image,
    # parametrized by the output of the body network
    stnet = mx.sym.SpatialTransformer(data=data_moving, loc=fc3, target_shape=image_shape, transform_type='affine',
                                      sampler_type="bilinear", name='SpatialTransformer')
    cor = mx.sym.Correlation(data1=data_fixed, data2=data_fixed, kernel_size=28, stride1=1, stride2=1, pad_size=0, max_displacement=0)
    cor2 = mx.sym.Correlation(data1=data_fixed, data2=data_moving, kernel_size=28, stride1=1, stride2=1, max_displacement=0)
    loss = mx.sym.MakeLoss(cor, normalization='batch')
    output = mx.sym.Group([mx.sym.BlockGrad(cor), mx.sym.BlockGrad(cor2), mx.sym.BlockGrad(fc3), loss])
    return output


def get_symbol(image_shape):
    return conv_net_regressor(image_shape)


def debug_symbol(sym):
    '''Get internals values for blobs (forward only).'''
    args = sym.list_arguments()
    output_names = []  # sym.list_outputs()

    sym = sym.get_internals()
    blob_names = sym.list_outputs()
    sym_group = []
    for i in range(len(blob_names)):
        if blob_names[i] not in args:
            x = sym[i]
            if blob_names[i] not in output_names:
                x = mx.symbol.BlockGrad(x, name=blob_names[i])
            sym_group.append(x)
    sym = mx.symbol.Group(sym_group)
    return sym


def custom_training_simple_bind(symbol, train_iter):
    '''
    Our own training method for the network. using the low-level simple_bind API
    Many code snippets are from https://github.com/apache/incubator-mxnet/blob/5ff545f2345f9b607b81546a168665bd63d02d9f/example/notebooks/simple_bind.ipynb
    :param symbol:
    :param train_iter:
    :return:
    '''

    # helper function
    def Init(key, arr):
        if "weight" in key:
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

    def customSGD(key, weight, grad, lr=0.1, grad_norm=1):
        # key is key for weight, we can customize update rule
        # weight is weight array
        # grad is grad array
        # lr is learning rate
        # grad_norm is scalar to norm gradient, usually it is batch_size
        norm = 1.0 / grad_norm
        # here we can bias' learning rate 2 times larger than weight
        if "weight" in key or "gamma" in key:
            weight[:] -= lr * (grad * norm)
        elif "bias" in key or "beta" in key:
            weight[:] -= 2.0 * lr * (grad * norm)
        else:
            pass

    # this should work, but it doesnt
    # shape = train_iter.provide_data#[0][1]  # gives us the data shape of one of the 2 data sources. Both have same shape
    # executor = symbol.simple_bind(ctx=mx.cpu(), data_shapes=shape, label_shapes=None)

    # luckily, this works
    executor = symbol.simple_bind(ctx=mx.cpu(), data_moving=(1, 1, 28, 28), data_fixed=(1, 1, 28, 28),
                                  label_shapes=None)

    # get argument arrays
    arg_arrays = executor.arg_arrays
    # get grad arrays
    grad_arrays = executor.grad_arrays
    # get aux_states arrays. Note: currently only BatchNorm symbol has auxiliary states, which is moving_mean and moving_var
    aux_arrays = executor.aux_arrays
    # get outputs from executor
    output_arrays = executor.outputs
    # The sequence of arrays is in same sequence of symbol arguments
    args = dict(zip(symbol.list_arguments(), arg_arrays))  # dict containing parameter names and values (i think)
    grads = dict(zip(symbol.list_arguments(), grad_arrays))
    outputs = dict(zip(symbol.list_outputs(), output_arrays))
    aux_states = dict(zip(symbol.list_auxiliary_states(), aux_arrays))

    # initialize parameters by uniform random numbers
    for key, arr in args.items():
        Init(key, arr)
    keys = symbol.list_arguments()
    # train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(5):
        train_iter.reset()
        avg_cor = 0
        i = 0
        for batch in train_iter:
            i += 1
            outs = executor.forward(is_train=True, data_fixed=batch.data[0], data_moving=batch.data[1])
            cor1 = executor.outputs[0]
            cor2 = executor.outputs[1]
            fc3 = executor.outputs[2]
            grad1 = executor.outputs[3]
            grad = executor.backward()  # compute gradients
            for key in keys:  # update parameters
                customSGD(key, args[key], grads[key])
            aval = cor1[0][0][0][0].asnumpy()[0]
            avg_cor += float(aval)
        print('Epoch %d, Training cor %s ' % (epoch, avg_cor/i))


def custom_training(mod, train_iter):
    shape = train_iter.provide_data  # [0][1]  # gives us the data shape of one of the 2 data sources. Both have same shape
    mod.bind(data_shapes=shape, label_shapes=None)
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))
    # use accuracy as the metric
    # metric = mx.metric.create('acc')
    # train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(5):
        train_iter.reset()
        # metric.reset()
        for batch in train_iter:
            cor = mod.forward(batch, is_train=True)  # compute predictions
            # mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            grad = mod.backward()  # compute gradients
            mod.update()  # update parameters
            print('Epoch %d, Training cor %s grad %s' % (epoch, cor, grad))


if __name__ == '__main__':
    mnist_shape = (1, 28, 28)
    iterators = get_mnist_data_iterator()
    net = get_symbol(mnist_shape)
    custom_training_simple_bind(net, iterators[0])
    # model = mx.mod.Module(symbol=net, context=mx.cpu(),
    #                       label_names=None, data_names=['data_fixed', 'data_moving'])
    # custom_training(model, iterators[0])
    # a = mx.viz.plot_network(net)
    # a.render()

    # model.fit(iterators[0],  # eval_data=val_iter,
    #                 optimizer='sgd',
    #                 optimizer_params={'learning_rate': 0.1},
    #                 eval_metric=mx.metric.Loss(),
    #                 num_epoch=10)
    # NOTE!
    # to get this to work right now with mxnet v. 0.12.1
    # You have to add the following lines at the beginning of the in update_metric function in executor_group.py!
    #
    # if self.label_layouts == None and self.label_names == None:
    #     # special case: We train without labels
    #     eval_metric.update_dict({}, {})
    #     return
