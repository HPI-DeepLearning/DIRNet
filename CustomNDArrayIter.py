
from __future__ import absolute_import
from collections import OrderedDict, namedtuple

import sys
import ctypes
import logging
import threading
try:
    import h5py
except ImportError:
    h5py = None
import numpy as np
#from .base import _LIB
#from .base import c_array, c_str, mx_uint, py_str
#from .base import DataIterHandle, NDArrayHandle
#from .base import mx_real_t
#from .base import check_call, build_param_doc as _build_param_doc
from mxnet.ndarray import NDArray
from mxnet.ndarray.sparse import CSRNDArray
#from .ndarray import _ndarray_cls
from mxnet.ndarray import array
from mxnet.ndarray import concatenate
import mxnet as mx
from mxnet.io import DataDesc
from mxnet.io import DataBatch
from mxnet.io import DataIter

# The only difference to the mxnet.io NDArrayIter is that provide_label returns None, as we do not have any labels
# This is needed to avoid an assertion error, see https://github.com/apache/incubator-mxnet/issues/8910
class NDArrayIter(DataIter):
    """Returns an iterator for ``mx.nd.NDArray``, ``numpy.ndarray``, ``h5py.Dataset``
    or ``mx.nd.sparse.CSRNDArray``.

    Example usage:
    ----------
    >>> data = np.arange(40).reshape((10,2,2))
    >>> labels = np.ones([10, 1])
    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='discard')
    >>> for batch in dataiter:
    ...     print batch.data[0].asnumpy()
    ...     batch.data[0].shape
    ...
    [[[ 36.  37.]
      [ 38.  39.]]
     [[ 16.  17.]
      [ 18.  19.]]
     [[ 12.  13.]
      [ 14.  15.]]]
    (3L, 2L, 2L)
    [[[ 32.  33.]
      [ 34.  35.]]
     [[  4.   5.]
      [  6.   7.]]
     [[ 24.  25.]
      [ 26.  27.]]]
    (3L, 2L, 2L)
    [[[  8.   9.]
      [ 10.  11.]]
     [[ 20.  21.]
      [ 22.  23.]]
     [[ 28.  29.]
      [ 30.  31.]]]
    (3L, 2L, 2L)
    >>> dataiter.provide_data # Returns a list of `DataDesc`
    [DataDesc[data,(3, 2L, 2L),<type 'numpy.float32'>,NCHW]]
    >>> dataiter.provide_label # Returns a list of `DataDesc`
    [DataDesc[softmax_label,(3, 1L),<type 'numpy.float32'>,NCHW]]

    In the above example, data is shuffled as `shuffle` parameter is set to `True`
    and remaining examples are discarded as `last_batch_handle` parameter is set to `discard`.

    Usage of `last_batch_handle` parameter:

    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='pad')
    >>> batchidx = 0
    >>> for batch in dataiter:
    ...     batchidx += 1
    ...
    >>> batchidx  # Padding added after the examples read are over. So, 10/3+1 batches are created.
    4
    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='discard')
    >>> batchidx = 0
    >>> for batch in dataiter:
    ...     batchidx += 1
    ...
    >>> batchidx # Remaining examples are discarded. So, 10/3 batches are created.
    3

    `NDArrayIter` also supports multiple input and labels.

    >>> data = {'data1':np.zeros(shape=(10,2,2)), 'data2':np.zeros(shape=(20,2,2))}
    >>> label = {'label1':np.zeros(shape=(10,1)), 'label2':np.zeros(shape=(20,1))}
    >>> dataiter = mx.io.NDArrayIter(data, label, 3, True, last_batch_handle='discard')

    `NDArrayIter` also supports ``mx.nd.sparse.CSRNDArray`` with `shuffle` set to `False`
    and `last_batch_handle` set to `discard`.

    >>> csr_data = mx.nd.array(np.arange(40).reshape((10,4))).tostype('csr')
    >>> labels = np.ones([10, 1])
    >>> dataiter = mx.io.NDArrayIter(csr_data, labels, 3, last_batch_handle='discard')
    >>> [batch.data[0] for batch in dataiter]
    [
    <CSRNDArray 3x4 @cpu(0)>,
    <CSRNDArray 3x4 @cpu(0)>,
    <CSRNDArray 3x4 @cpu(0)>]

    Parameters
    ----------
    data: array or list of array or dict of string to array
        The input data.
    label: array or list of array or dict of string to array, optional
        The input label.
    batch_size: int
        Batch size of data.
    shuffle: bool, optional
        Whether to shuffle the data.
        Only supported if no h5py.Dataset inputs are used.
    last_batch_handle : str, optional
        How to handle the last batch. This parameter can be 'pad', 'discard' or
        'roll_over'. 'roll_over' is intended for training and can cause problems
        if used for prediction.
    data_name : str, optional
        The data name.
    label_name : str, optional
        The label name.
    """
    def __init__(self, data, label=None, batch_size=1, shuffle=False,
                 last_batch_handle='pad', data_name='data',
                 label_name='softmax_label'):
        super(NDArrayIter, self).__init__(batch_size)

        self.data = mx.io._init_data(data, allow_empty=False, default_name=data_name)
        self.label = mx.io._init_data(label, allow_empty=True, default_name=label_name)
        if isinstance(data, CSRNDArray) or isinstance(label, CSRNDArray):
            assert(shuffle is False), \
                  "`NDArrayIter` only supports ``CSRNDArray`` with `shuffle` set to `False`"
            assert(last_batch_handle == 'discard'), "`NDArrayIter` only supports ``CSRNDArray``" \
                                                    " with `last_batch_handle` set to `discard`."

        self.idx = np.arange(self.data[0][1].shape[0])
        # shuffle data
        if shuffle:
            np.random.shuffle(self.idx)
            self.data = [(k, array(v.asnumpy()[self.idx], v.context))
                         if not (isinstance(v, h5py.Dataset)
                                 if h5py else False) else (k, v)
                         for k, v in self.data]
            self.label = [(k, array(v.asnumpy()[self.idx], v.context))
                          if not (isinstance(v, h5py.Dataset)
                                  if h5py else False) else (k, v)
                          for k, v in self.label]

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data[0][1].shape[0] - self.data[0][1].shape[0] % batch_size
            self.idx = self.idx[:new_n]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)
        self.num_data = self.idx.shape[0]
        assert self.num_data >= batch_size, \
            "batch_size needs to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator."""
        return [
            DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype)
            for k, v in self.data
        ]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        return None

    def hard_reset(self):
        """Ignore roll over data and set to start."""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only."""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [
                # np.ndarray or NDArray case
                x[1][self.cursor:self.cursor + self.batch_size]
                if isinstance(x[1], (np.ndarray, NDArray)) else
                # h5py (only supports indices in increasing order)
                array(x[1][sorted(self.idx[
                    self.cursor:self.cursor + self.batch_size])][[
                        list(self.idx[self.cursor:
                                      self.cursor + self.batch_size]).index(i)
                        for i in sorted(self.idx[
                            self.cursor:self.cursor + self.batch_size])
                    ]]) for x in data_source
            ]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [
                # np.ndarray or NDArray case
                concatenate([x[1][self.cursor:], x[1][:pad]])
                if isinstance(x[1], (np.ndarray, NDArray)) else
                # h5py (only supports indices in increasing order)
                concatenate([
                    array(x[1][sorted(self.idx[self.cursor:])][[
                        list(self.idx[self.cursor:]).index(i)
                        for i in sorted(self.idx[self.cursor:])
                    ]]),
                    array(x[1][sorted(self.idx[:pad])][[
                        list(self.idx[:pad]).index(i)
                        for i in sorted(self.idx[:pad])
                    ]])
                ]) for x in data_source
            ]

    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0

