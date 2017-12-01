import logging
logging.getLogger().setLevel(logging.DEBUG)
import mxnet as mx
import numpy as np
mnist = mx.test_utils.get_mnist()

batch_size = 100
weighted_train_labels =np.zeros((mnist['train_label'].shape[0],np.max(mnist['train_label'])+ 1))
weighted_train_labels[np.arange(mnist['train_label'].shape[0]),mnist['train_label']] = 1
train_iter = mx.io.NDArrayIter(mnist['train_data'], {'label':weighted_train_labels}, batch_size, shuffle=True)

weighted_test_labels = np.zeros((mnist['test_label'].shape[0],np.max(mnist['test_label'])+ 1))
weighted_test_labels[np.arange(mnist['test_label'].shape[0]),mnist['test_label']] = 1
val_iter = mx.io.NDArrayIter(mnist['test_data'], {'label':weighted_test_labels}, batch_size)

data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
#lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

label = mx.sym.var('label')
correlation=mx.symbol.Correlation(data,data)
correlation_output = mx.sym.BlockGrad(data = correlation,name = 'correlation')
ce = -mx.sym.sum(mx.sym.broadcast_mul(fc2,label),1)
lenet = mx.symbol.MakeLoss(ce, normalization='batch')

sym = mx.sym.Group([softmax_output,lenet])

def custom_metric(label,softmax):
    return len(np.where(np.argmax(softmax,1)==np.argmax(label,1))[0])/float(label.shape[0])

eval_metrics = mx.metric.CustomMetric(custom_metric,name='custom-accuracy', output_names=['softmax_output'],label_names=['label'])

lenet_model = mx.mod.Module(symbol=sym, context=mx.gpu(),data_names=['data'], label_names=['label'])
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric=eval_metrics,#mx.metric.Loss(),#'acc',
                #batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)
