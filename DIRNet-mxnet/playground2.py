'''
Implementation of the following paper:
https://arxiv.org/abs/1704.06065
'''
import logging

from requests.api import post

logging.getLogger().setLevel(logging.DEBUG)
import mxnet as mx
import numpy as np
import helper as hlp
#import CustomNDArrayIter as customIter

def conv_net_regressor(image_shape, bn_mom=0.9):
    # We have 2 data sources and concatenate them
    data_fixed = mx.sym.Variable(name='data')
    #data_moving = mx.sym.Variable(name='data_moving')
    #concat_data = mx.sym.concat(*[data_fixed, data_moving])
    batched = mx.sym.BatchNorm(data=data_fixed, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
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
        if i == 3:
            body_out = body
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn' + str(i))
        if i == 3:
            body_out2 = body
        # TO DO: the original authors use exponential linear units as activation
        body = mx.sym.LeakyReLU(data=body, act_type='leaky', name='relu' + str(i))
        body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(1, 1), pad=(1, 1), pool_type='avg')
    # Subsequently, three 1 × 1 convolutional layers are applied to make the ConvNet regressor fully convolutional
    #for k in range(3):
    i = i + 4
    a_conv = mx.sym.Convolution(data=body, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                              no_bias=True, name="conv" + str(i))
    body = mx.sym.BatchNorm(data=a_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn' + str(i))
    # TO DO: the original authors use exponential linear units as activation
    body = mx.sym.LeakyReLU(data=body, act_type='leaky', name='relu' + str(i))
   # body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='avg')

    flatten = mx.sym.flatten(data=body)
    fc3 = mx.sym.FullyConnected(data=flatten, num_hidden=20)
    net = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    output = mx.sym.Group([mx.sym.BlockGrad(body_out), mx.sym.BlockGrad(body_out2), net])
    return output


def get_symbol(image_shape):
    return conv_net_regressor(image_shape)


def custom_training_simple_bind(symbol, iterators):
    '''
    Our own training method for the network. using the low-level simple_bind API
    Many code snippets are from https://github.com/apache/incubator-mxnet/blob/5ff545f2345f9b607b81546a168665bd63d02d9f/example/notebooks/simple_bind.ipynb
    :param symbol:
    :param train_iter:
    :return:
    '''

    # helper function
    def Init(key, arr):
        # if "fullyconnected0_bias" in key:
        #     # initialize with identity transformation
        #     initial = np.array([[1., 0, 0], [0, 1., 0]])
        #     initial = initial.astype('float32').flatten()
        #     arr[:] = initial
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

    def customSGD(key, weight, grad, lr=0.01, grad_norm=1):
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

    def Accuracy(label, pred_prob):
        pred = np.argmax(pred_prob, axis=1)
        return np.sum(label == pred) * 1.0 / label.shape[0]

    executor = symbol.simple_bind(ctx=mx.cpu(), data=(1, 1, 28, 28),
                                  label_shapes=None, grad_req='write')

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
    train_iter = iterators
    pred_prob = mx.nd.zeros(executor.outputs[2].shape)
    # train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(50):
        train_iter.reset()
        avg_cor = 0
        i = 0
        train_acc = 0.
        fc3 = None
        #fixed_img_data = train_iter.next().data
        for batch in train_iter:
            i += 1
            # printNumpyArray(batch.data[0][0][0])
            label = batch.label[0]
            outs = executor.forward(is_train=True, data=batch.data[0], softmax_label=label)
            # pre_bn = executor.outputs[0]
            # post_bn = executor.outputs[1]
            # shape_pre = pre_bn.shape
            # shape_post = post_bn.shape
            # summed_pre = np.sum(pre_bn.asnumpy())
            # summed_post = np.sum(post_bn.asnumpy())
            # pure_bn = hlp.pure_batch_norm(X=pre_bn, gamma=args['bn3_gamma'], beta=args['bn3_beta'])
            # fc3 = executor.outputs[1]
            pred_prob[:] = executor.outputs[2]
            train_acc += Accuracy(label.asnumpy(), pred_prob.asnumpy())
           #  stnet = executor.outputs[1]
           #  fc3 = executor.outputs[2]
           # # print("Affine transformation parameters Theta: " + str(fc3))
           #  loss = executor.outputs[3]

            executor.backward()  # compute gradients
            if i%1800 == 0:
                print("batch " + str(i))    
                hlp.printNontZeroGradients(grads)
                #print(grads['conv3_weight'])
                print(args['conv3_weight'][0])
            for key in keys:  # update parameters
                customSGD(key, args[key], grads[key])
            # aval = cor1[0][0][0][0].asnumpy()[0]
            # avg_cor += float(aval)
        # print("Affine transformation parameters Theta: " + str(fc3))
        print('Epoch %d, Training acc %s ' % (epoch, train_acc/i))


if __name__ == '__main__':
    mnist_shape = (1, 1, 28, 28)
    mnist = hlp.get_mnist(mnistdir='/home/adrian/PycharmProjects/DIRNet/data/')  # or use mnist = mx.test_utils.get_mnist() to download
    #mnist = mx.test_utils.get_mnist()
    standard_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], 1, shuffle=True)
    batch_size = 1
    iterator = hlp.get_mnist_data_iterator(mnistdir='/home/adrian/PycharmProjects/DIRNet/data/', digit=1)
    net = get_symbol(mnist_shape)
    custom_training_simple_bind(net, standard_iter)
