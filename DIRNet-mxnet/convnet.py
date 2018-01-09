'''
Implementation of the following paper:
https://arxiv.org/abs/1704.06065
'''
import logging
logging.getLogger().setLevel(logging.DEBUG)
import mxnet as mx
import numpy as np
import os
import helper as hlp
import RegistrationIterator as RIter


def conv_net_regressor(shape, use_additional_pool=False, bn_mom=0.9):
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
        body = mx.sym.LeakyReLU(data=body, act_type='elu', name='relu' + str(i))
        body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(1, 1), pad=(1, 1), pool_type='avg')
    # Subsequently, three 1 × 1 convolutional layers are applied to make the ConvNet regressor fully convolutional
    for k in range(2):
        i = k + 4
        body = mx.sym.Convolution(data=body, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                  no_bias=True, name="conv" + str(i))
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn' + str(i))
        # body = mx.sym.Activation(data=body, act_type='relu', name='relu' + str(i))
        body = mx.sym.LeakyReLU(data=body, act_type='elu', name='relu' + str(i))
        if use_additional_pool:
            body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='avg')
    #  body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='avg')
    flatten = mx.sym.flatten(data=body)
    fc3 = mx.sym.FullyConnected(data=flatten, num_hidden=6)
    fc3 = mx.sym.Activation(data=fc3, act_type='tanh', name='tanh_after_fc')
    # The Spatial Transformer performs a affine transformation to the moving image,
    # parametrized by the output of the body network
    stnet = mx.sym.SpatialTransformer(data=data_moving, loc=fc3, target_shape=(shape[2], shape[3]), transform_type='affine',
                                      sampler_type="bilinear", name='SpatialTransformer')
    #cor = mx.sym.Correlation(data1=stnet, data2=data_fixed, kernel_size=28, stride1=2, stride2=2, pad_size=0, max_displacement=0)
    #cor2 = mx.sym.Correlation(data1=data_fixed, data2=stnet, kernel_size=28, stride1=1, stride2=1, max_displacement=0)
    # loss = mx.sym.MakeLoss(hlp.ncc(stnet, data_fixed), normalization='batch')
    loss = mx.sym.MakeLoss(hlp.rmse(stnet, data_fixed), normalization='batch')
    output = mx.sym.Group([mx.sym.BlockGrad(fc3), mx.sym.BlockGrad(stnet), mx.sym.BlockGrad(fc3), loss])
    return output


def custom_training_simple_bind(symbol, iterators, ctx=mx.gpu(), epochs=10, save_path='./', lr=0.0000001):
    '''
    Our own training method for the network. using the low-level simple_bind API
    Many code snippets are from https://github.com/apache/incubator-mxnet/blob/5ff545f2345f9b607b81546a168665bd63d02d9f/example/notebooks/simple_bind.ipynb
    :param symbol:
    :param train_iter:
    :return:
    '''

    # helper function
    def Init(key, arr):
        if "fullyconnected0_weight" in key:
            # initialize with identity transformation
            arr[:] = 0
        elif "fullyconnected0_bias" in key:
            # initialize with identity transformation
            initial = np.array([[1., 0, 0], [0, 1., 0]])
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

    def customSGD(key, weight, grad, lr=lr, grad_norm=1):
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

    executor = symbol.simple_bind(ctx=ctx, data_moving=(1, 1, 28, 28), data_fixed=(1, 1, 28, 28),
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
    args = dict(zip(symbol.list_arguments(), arg_arrays))  # dict containing parameter names and values
    grads = dict(zip(symbol.list_arguments(), grad_arrays))
    outputs = dict(zip(symbol.list_outputs(), output_arrays))
    aux_states = dict(zip(symbol.list_auxiliary_states(), aux_arrays))

    # initialize parameters by uniform random numbers
    #mx.random.seed(2)
    for key, arr in args.items():
        Init(key, arr)
    keys = symbol.list_arguments()
    train_iter = iterators[0]
    eval_iter = iterators[1]
    debug = False
    # train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(epochs):
        train_iter.reset()
        fc3 = None
        avg_cor = 0
        i = 0
        fixed_img_data = train_iter.next().data
        #hlp.printNumpyArray(fixed_img_data[0][0][0], thresh=0)
        for batch in train_iter:
            i += 1
            executor.forward(is_train=True, data_fixed=fixed_img_data[0], data_moving=batch.data[0])
            cor1 = executor.outputs[0]
            stnet = executor.outputs[1]
            loss = executor.outputs[3]
            fc3 = executor.outputs[2]
            if debug:
                if np.sum(stnet.asnumpy()) == 0:
                    print('   STN produces empty feature map!')
                else:
                    print('   STN seems to work')
                #sh = stnet.shape
                print("Affine transformation parameters Theta: " + str(fc3))
                print("loss " + str(loss.asnumpy()[0]))
                #hlp.printNumpyArray(stnet.asnumpy()[0][0], thresh=0)
                hlp.printNontZeroGradients(grads)
            #if loss != -1.0:  # otherwise ncc gradient is NaN
            executor.backward()  # compute gradients
            for key in keys:  # update parameters
                customSGD(key, args[key], grads[key])
            aval = loss.asnumpy()[0]
            avg_cor += float(aval)
        print("Affine transformation parameters Theta: " + str(fc3))
        print('Epoch %d, Training avg rmse %s ' % (epoch, avg_cor/i))
        eval_iter.reset()
        avg_cor = 0
        i = 0
        for batch in eval_iter:
            executor.forward(is_train=True, data_fixed=fixed_img_data[0], data_moving=batch.data[0])
            loss = executor.outputs[3]
            aval = loss.asnumpy()[0]
            avg_cor += float(aval)
            i += 1
        print('Epoch %d, Evaluation avg rmse %s ' % (epoch, avg_cor/i))

    # # save some to see what the model does
    # hlp.saveArrayAsImg(fixed_img_data[0].asnumpy()[0][0], "./fixed.png")
    # train_iter.reset()
    # for l in range(7):
    #     batch = train_iter.next()
    #     moving = batch.data[0]
    #     executor.forward(is_train=True, data_fixed=fixed_img_data[0], data_moving=moving)
    #     stnet = executor.outputs[1]
    #     hlp.saveArrayAsImg(stnet.asnumpy()[0][0], './{0}0warped_{0}.png'.format(l))
    #     hlp.saveArrayAsImg(moving.asnumpy()[0][0], './{0}1original_{0}.png'.format(l))
    return executor


def cardiac_training(symbol, img_shape, data, ctx=mx.gpu(), epochs=10, lr=0.00000000001):
    '''
    Our own training method for the network. using the low-level simple_bind API
    Many code snippets are from https://github.com/apache/incubator-mxnet/blob/5ff545f2345f9b607b81546a168665bd63d02d9f/example/notebooks/simple_bind.ipynb
    nearly the same as training on mnist, just different iterators
    :param symbol:
    :param train_iter:
    :return:
    '''

    # helper function
    def Init(key, arr):
        if "fullyconnected0_weight" in key:
            # initialize with identity transformation
            arr[:] = 0
        elif "fullyconnected0_bias" in key:
            # initialize with identity transformation
            initial = np.array([[1., 0, 0], [0, 1., 0]])
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

    def customSGD(key, weight, grad, lr=lr, grad_norm=1):
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

    executor = symbol.simple_bind(ctx=ctx, data_moving=img_shape, data_fixed=img_shape,
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
    args = dict(zip(symbol.list_arguments(), arg_arrays))  # dict containing parameter names and values
    grads = dict(zip(symbol.list_arguments(), grad_arrays))
    outputs = dict(zip(symbol.list_outputs(), output_arrays))
    aux_states = dict(zip(symbol.list_auxiliary_states(), aux_arrays))

    # initialize parameters by uniform random numbers
    #mx.random.seed(2)
    for key, arr in args.items():
        Init(key, arr)
    keys = symbol.list_arguments()
    debug = True
    data_fixed = np.empty((1, 1, img_shape[2], img_shape[3]))
    data_mov = np.empty((1, 1, img_shape[2], img_shape[3]))
    for epoch in range(epochs):
        fc3 = None
        avg_cor = 0
        i = 0
        for batch in data:
            i += 1
            print(i)
            data_fixed[0][0] = batch[0]
            data_mov[0][0] = batch[1]
            executor.forward(is_train=True, data_fixed=data_fixed, data_moving=data_mov)
            stnet = executor.outputs[1]
            loss = executor.outputs[3]
            fc3 = executor.outputs[2]
            if debug:
                if np.sum(stnet.asnumpy()) == 0:
                    print('   STN produces empty feature map!')
                else:
                    print('   STN seems to work')
                #sh = stnet.shape
                print("Affine transformation parameters Theta: " + str(fc3))
                print("loss " + str(loss.asnumpy()[0]))
                #hlp.printNumpyArray(stnet.asnumpy()[0][0], thresh=0)
                hlp.printNontZeroGradients(grads)
            #if loss != -1.0:  # otherwise ncc gradient is NaN
            executor.backward()  # compute gradients
            for key in keys:  # update parameters
                customSGD(key, args[key], grads[key])
            aval = loss.asnumpy()[0]
            avg_cor += float(aval)
        print("Affine transformation parameters Theta: " + str(fc3))
        print('Epoch %d, Training avg rmse %s ' % (epoch, avg_cor/i))
        # eval_iter.reset()
        # avg_cor = 0
        # i = 0
        # for batch in eval_iter:
        #     executor.forward(is_train=True, data_fixed=fixed_img_data[0], data_moving=batch.data[0])
        #     loss = executor.outputs[3]
        #     aval = loss.asnumpy()[0]
        #     avg_cor += float(aval)
        #     i += 1
        # print('Epoch %d, Evaluation avg rmse %s ' % (epoch, avg_cor/i))

    # # save some to see what the model does
    # hlp.saveArrayAsImg(fixed_img_data[0].asnumpy()[0][0], "./fixed.png")
    # train_iter.reset()
    # for l in range(7):
    #     batch = train_iter.next()
    #     moving = batch.data[0]
    #     executor.forward(is_train=True, data_fixed=fixed_img_data[0], data_moving=moving)
    #     stnet = executor.outputs[1]
    #     hlp.saveArrayAsImg(stnet.asnumpy()[0][0], './{0}0warped_{0}.png'.format(l))
    #     hlp.saveArrayAsImg(moving.asnumpy()[0][0], './{0}1original_{0}.png'.format(l))
    return executor


def save_params(symbol, executor, path='./dirnet_params.json'):

    # get argument arrays
    arg_arrays = executor.arg_arrays
    # get aux_states arrays. Note: currently only BatchNorm symbol has auxiliary states, which is moving_mean and moving_var
    aux_arrays = executor.aux_arrays
    # The sequence of arrays is in same sequence of symbol arguments
    args = dict(zip(symbol.list_arguments(), arg_arrays))  # dict containing parameter names and values
    aux_states = dict(zip(symbol.list_auxiliary_states(), aux_arrays))
    save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in args.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_states.items()})
    mx.nd.save(os.path.abspath(path), save_dict)
    #mx.nd.save(path, [arg_arrays, aux_arrays])


def load_params_to_exec(symbol, ctx=mx.gpu(), path='./dirnet_params.json'):
    save_dict = mx.nd.load(os.path.abspath(path))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    executor = symbol.simple_bind(ctx=ctx, data_moving=(1, 1, 28, 28), data_fixed=(1, 1, 28, 28),
                                  label_shapes=None, grad_req='null')
    executor.copy_params_from(arg_params=arg_params, aux_params=aux_params)
    return executor


def predict(executor, iterator):
    avg_cor = 0
    i = 0
    fixed_img_data = iterator.next().data
    #hlp.printNumpyArray(fixed_img_data[0][0][0], thresh=0)
    for batch in iterator:
        i += 1
        executor.forward(is_train=True, data_fixed=fixed_img_data[0], data_moving=batch.data[0])
        cor1 = executor.outputs[0]
        stnet = executor.outputs[1]
        loss = executor.outputs[3]
        fc3 = executor.outputs[2]
        aval = loss.asnumpy()[0]
        avg_cor += float(aval)
    print("Affine transformation parameters Theta: " + str(fc3))


if __name__ == '__main__':
    cardio_shape = (222, 247)
    ctx = mx.gpu()
    # mnist_shape = (1, 1, 28, 28)
    # net = get_symbol(mnist_shape)
    #mnist = get_mnist(mnistdir='./data/')  # or use mnist = mx.test_utils.get_mnist() to download
    #iterator = hlp.get_mnist_data_iterator(mnistdir='./data/', digit=0)
    # trained_exec = custom_training_simple_bind(symbol=net, epochs=1, ctx=ctx, iterators=iterator)
    # iter = RIter.RegistrationIter(ES_path='/home/adrian/Documents/dl2/Cardiac/ES',
    #              ED_path='/home/adrian/Documents/dl2/Cardiac/ED', shape=cardio_shape)
    # iter.reset()
    # for batch_fixed in iter:
    #    # print(batch_fixed)
    #    # print(str(np.shape(batch_fixed.data)))
    #     print(str(np.shape(batch_fixed.data[0])))
    net = conv_net_regressor(shape=(1, 1, cardio_shape[0], cardio_shape[1]), use_additional_pool=True)
    # hlp.create_imglist('/home/adrian/Documents/dl2/Cardiac/ES')
    # hlp.create_imglist('/home/adrian/Documents/dl2/Cardiac/ED')
    # The following does not work with greyscale images...
    # iterator_ES = mx.image.ImageIter(batch_size=1, data_shape=cardio_shape,
    #                                  path_root='/home/adrian/Documents/dl2/Cardiac/ES',
    #                                  path_imglist='/home/adrian/Documents/dl2/Cardiac/ES/imglist.txt')
    # iterator_ED = mx.image.ImageIter(batch_size=1, data_shape=cardio_shape,
    #                                  path_root='/home/adrian/Documents/dl2/Cardiac/ED',
    #                                  path_imglist='/home/adrian/Documents/dl2/Cardiac/ES/imglist.txt')

    data = hlp.read_cardio_dirs_to_ndarray(path_moving='/home/adrian/Documents/dl2/Cardiac/ES',
                 path_fixed='/home/adrian/Documents/dl2/Cardiac/ED', shape=cardio_shape)
    trained_exec = cardiac_training(symbol=net, img_shape=(1, 1, cardio_shape[0], cardio_shape[1]),
                                    epochs=1, ctx=ctx, data=data)
    save_params(executor=trained_exec, symbol=net)
    loaded_exec = load_params_to_exec(net, ctx=ctx)
