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
from os import listdir
from os.path import isfile, join
import scipy.ndimage as ndimage
import scipy.misc as misc


# The only difference to the mxnet.io NDArrayIter is that provide_label returns None, as we do not have any labels
# This is needed to avoid an assertion error, see https://github.com/apache/incubator-mxnet/issues/8910
class RegistrationIter(DataIter):

    def find_moving_img(self, arr, start_idx, fixed_name):
        # patient035_frame01.nz.10.png
        patient_id = fixed_name[7:10]
        slice_id = fixed_name[22:24]
        for i in range(len(arr)):
            idx = (i + start_idx) % len(arr)  # iterate through the whole array but dont start at 0
            moving_name = arr[idx]
            if patient_id == moving_name[7:10] and slice_id == moving_name[22:24]:
                return moving_name
        return None

    def read_cardio_dirs_to_ndarray(self, path_fixed, path_moving, shape):
        onlyfiles_fixed = [f for f in listdir(path_fixed) if isfile(join(path_fixed, f))]
        onlyfiles_moving = [f for f in listdir(path_moving) if isfile(join(path_moving, f))]
        # out_fix = np.empty(shape=(shape[1], shape[2]))
        # out_mov = np.empty(shape=(shape[1], shape[2]))
        arrays_fix = []
        arrays_mov = []
        for i, fixed in enumerate(onlyfiles_fixed):
            if fixed.endswith('.png'):
                moving = self.find_moving_img(onlyfiles_moving, i, fixed)
                assert moving is not None

                abspath = join(path_fixed, fixed)
                pic_fix = ndimage.imread(abspath, flatten=True)
                pic_fix = misc.imresize(pic_fix, (shape[1], shape[2]))

                abspath = join(path_moving, moving)
                pic_mov = ndimage.imread(abspath, flatten=True)
                pic_mov = misc.imresize(pic_mov, (shape[1], shape[2]))
                arrays_fix.append(np.stack([pic_fix, pic_mov]))
                #arrays_mov.append(pic_mov)
        return arrays_fix
        # sh = np.shape(arrays_fix)
        # out = np.stack(arrays_fix)
        # sh2 = np.shape(out)
        # return (np.stack(arrays_fix), np.stack(arrays_mov))
       # return (np.stack(arrays_fix), np.stack(arrays_mov))

    """Returns an iterator for ``mx.nd.NDArray``, ``numpy.ndarray``, ``h5py.Dataset``
    or ``mx.nd.sparse.CSRNDArray``.
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
    def __init__(self, ES_path, ED_path, shape, batch_size=1, shuffle=False,
                 last_batch_handle='pad'):
        super(RegistrationIter, self).__init__(batch_size)

        data = self.read_cardio_dirs_to_ndarray(path_fixed=ED_path, path_moving=ES_path, shape=shape)
        self.data = mx.io._init_data(data, allow_empty=False, default_name='data')
        #self.data_moving = mx.io._init_data(data[1], allow_empty=False, default_name='data_fixed')

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

       # self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        #self.num_source = len(self.data_list)
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
        return None

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0