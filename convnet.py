'''
Implementation of the following paper:
https://arxiv.org/abs/1704.06065
'''

import mxnet as mx
import numpy as np


def conv_net_regressor(image_shape, bn_mom=0.9):
    (nchannel, height, width) = image_shape
    # We have 2 data sources and concatenate them
    data_fixed = mx.sym.Variable(name='data_fixed')
    data_moving = mx.sym.Variable(name='data_moving')
    concat_data = mx.sym.concat(*[data_fixed, data_moving])
    data = mx.sym.BatchNorm(data=concat_data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    # The number of kernels per layer can be of arbitrary size, but the number of kernels of the output layer is
    # determined by the dimensionality of the input images
    filter_list = [16, 64, 128, 256]
    # four alternating layers of 3 × 3 convolutions with 0-padding and 2 × 2 downsampling layers
    for i in range(4):
        body = mx.sym.Convolution(data=data, num_filter=filter_list[i], kernel=(3, 3), stride=(2, 2), pad=(0, 0),
                                  no_bias=True, name="conv" + str(i))
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn' + str(i))
        # TO DO: the original authors use exponential linear units as activation
        body = mx.sym.Activation(data=body, act_type='relu', name='relu' + str(i))
        body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='avg')
    # Subsequently, three 1 × 1 convolutional layers are applied to make the ConvNet regressor fully convolutional
    for k in range(3):
        i = k+4
        body = mx.sym.Convolution(data=body, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                  no_bias=True, name="conv" + str(i))
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn' + str(i))
        # TO DO: the original authors use exponential linear units as activation
        body = mx.sym.Activation(data=body, act_type='relu', name='relu' + str(i))
        body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='avg')

    # Nicht sicher ob loc einfach der output vom convnet regressor ist.
    # Ich vermute man muss die  anzahl filter und kernel sizes so anpassen dass das irgendwie zusammen passt
    body = mx.sym.SpatialTransformer(data=data_moving, loc=body)

def get_symbol(image_shape):
    return conv_net_regressor(image_shape)
